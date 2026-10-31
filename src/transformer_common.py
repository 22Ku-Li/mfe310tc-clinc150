"""Shared data, inference, and output utilities for Transformer experiments."""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_config, resolve_project_path
from src.data import PROJECT_ROOT
from src.evaluate import compute_classification_metrics, save_classification_report, save_top_confusions

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache/huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def set_reproducible_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch, including all visible CUDA devices."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def resolve_device(requested: str) -> torch.device:
    """Resolve auto/cpu/cuda and reject a missing explicitly requested GPU."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this environment")
    if requested not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported device: {requested}")
    return torch.device(requested)


def load_and_validate_split(path: Path) -> pd.DataFrame:
    """Load one processed split and verify the prediction-output contract."""
    frame = pd.read_csv(path)
    required = {"sample_id", "text", "label"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path} lacks required columns: {sorted(required - set(frame.columns))}")
    if frame["sample_id"].duplicated().any():
        raise ValueError(f"Duplicate sample IDs in {path}")
    if frame[["text", "label"]].isna().any().any():
        raise ValueError(f"Missing text or labels in {path}")
    return frame


class TokenizedIntentDataset(Dataset):
    """A fixed, eagerly tokenized utterance dataset with integer labels."""

    def __init__(self, frame: pd.DataFrame, tokenizer, label_to_id: dict[str, int], max_length: int):
        self.encodings = tokenizer(
            frame["text"].tolist(),
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor([label_to_id[label] for label in frame["label"]], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {key: value[index] for key, value in self.encodings.items()}
        item["labels"] = self.labels[index]
        return item


def seed_worker(worker_id: int) -> None:
    """Seed each DataLoader worker from PyTorch's deterministic worker seed."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_data_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    """Create a DataLoader with an explicit, reproducible shuffle generator."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator,
    )


def move_inputs_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> tuple[dict, torch.Tensor]:
    """Move model inputs and labels separately so loss computation stays explicit."""
    labels = batch["labels"].to(device)
    inputs = {
        key: batch[key].to(device)
        for key in ("input_ids", "attention_mask")
        if key in batch
    }
    return inputs, labels


def autocast_context(device: torch.device, enabled: bool):
    """Use FP16 autocast only when CUDA mixed precision is enabled."""
    return torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=enabled and device.type == "cuda",
    )


@torch.no_grad()
def predict_probabilities(
    model,
    data_loader: DataLoader,
    device: torch.device,
    mixed_precision: bool,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Run single-forward evaluation and return labels, probabilities, CE, and latency."""
    model.eval()
    all_labels = []
    all_probabilities = []
    cross_entropy_sum = 0.0
    row_count = 0
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for batch in data_loader:
        inputs, labels = move_inputs_to_device(batch, device)
        with autocast_context(device, mixed_precision):
            logits = model(**inputs).logits
        logits = logits.float()
        cross_entropy_sum += F.cross_entropy(logits, labels, reduction="sum").item()
        probabilities = F.softmax(logits, dim=-1)
        all_labels.append(labels.cpu().numpy())
        all_probabilities.append(probabilities.cpu().numpy())
        row_count += len(labels)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return (
        np.concatenate(all_labels),
        np.concatenate(all_probabilities),
        cross_entropy_sum / row_count,
        elapsed,
    )


def save_evaluation_outputs(
    frame: pd.DataFrame,
    labels: list[str],
    true_ids: np.ndarray,
    probabilities: np.ndarray,
    output_dir: Path,
    oos_label: str,
    extra_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Save metrics and row-level evidence using the same contract as M0."""
    output_dir.mkdir(parents=True, exist_ok=True)
    predicted_ids = probabilities.argmax(axis=1)
    id_to_label = {index: label for index, label in enumerate(labels)}
    true_labels = [id_to_label[int(index)] for index in true_ids]
    predicted_labels = [id_to_label[int(index)] for index in predicted_ids]
    label_to_id = {label: index for index, label in enumerate(labels)}
    oos_index = label_to_id[oos_label]
    oos_scores = probabilities[:, oos_index]

    metrics = compute_classification_metrics(
        true_labels,
        predicted_labels,
        labels=labels,
        oos_label=oos_label,
        oos_scores=oos_scores,
    )
    if extra_metrics:
        metrics.update(extra_metrics)

    prediction_frame = frame[["sample_id", "text", "label"]].copy()
    prediction_frame = prediction_frame.rename(columns={"label": "true_label"})
    prediction_frame["predicted_label"] = predicted_labels
    prediction_frame["predicted_probability"] = probabilities[
        np.arange(len(predicted_ids)), predicted_ids
    ]
    prediction_frame["true_label_probability"] = probabilities[
        np.arange(len(true_ids)), true_ids
    ]
    prediction_frame["oos_score"] = oos_scores
    prediction_frame["correct"] = prediction_frame["true_label"] == prediction_frame["predicted_label"]
    prediction_frame.to_csv(output_dir / "predictions.csv", index=False)

    save_classification_report(
        true_labels, predicted_labels, labels, output_dir / "per_intent_metrics.csv"
    )
    save_top_confusions(
        true_labels, predicted_labels, labels, output_dir / "top_confusions.csv"
    )
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
