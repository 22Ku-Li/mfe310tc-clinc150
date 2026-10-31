"""Evaluate a frozen Transformer checkpoint without retraining or model selection."""

from __future__ import annotations

import argparse
import json
import platform
import sys

import numpy as np
import pandas as pd
import torch
import yaml

from src.data import PROJECT_ROOT, sha256
from src.transformer_common import (
    TokenizedIntentDataset,
    load_and_validate_split,
    make_data_loader,
    predict_probabilities,
    resolve_device,
    resolve_project_path,
    save_evaluation_outputs,
    set_reproducible_seed,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to a frozen best checkpoint")
    parser.add_argument("--split", required=True, choices=["val", "test"])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    parser.add_argument("--allow-test", action="store_true")
    args = parser.parse_args()
    if args.split == "test" and not args.allow_test:
        parser.error("Test evaluation requires --allow-test after all model choices are frozen")

    try:
        import transformers
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as error:
        raise SystemExit("Install Transformer dependencies with pip install -r requirements-transformers.txt") from error

    checkpoint = resolve_project_path(args.checkpoint)
    state_path = checkpoint / "training_state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing training state: {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    config = state["config"]
    labels = state["labels"]
    label_to_id = {label: index for index, label in enumerate(labels)}
    seed = int(state["seed"])
    requested_device = args.device or config["training"]["device"]
    device = resolve_device(requested_device)
    set_reproducible_seed(seed, deterministic=config["training"]["deterministic"])

    data_path = resolve_project_path(config["data"][f"{args.split}_path"])
    frame = load_and_validate_split(data_path)
    if set(frame["label"]) != set(labels):
        raise ValueError("Evaluation labels differ from the checkpoint's 151-class label set")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, local_files_only=True)
    model.to(device)
    dataset = TokenizedIntentDataset(frame, tokenizer, label_to_id, config["model"]["max_length"])
    data_loader = make_data_loader(
        dataset,
        config["training"]["batch_size"],
        False,
        seed,
        config["training"]["num_workers"],
        device.type == "cuda",
    )
    use_amp = bool(config["training"]["mixed_precision"] and device.type == "cuda")
    true_ids, probabilities, cross_entropy, inference_seconds = predict_probabilities(
        model, data_loader, device, use_amp
    )

    experiment = state["experiment_name"]
    result_dir = PROJECT_ROOT / "results" / experiment / f"seed_{seed}" / args.split
    metrics = save_evaluation_outputs(
        frame,
        labels,
        true_ids,
        probabilities,
        result_dir,
        config["evaluation"]["oos_label"],
        extra_metrics={
            "experiment": experiment,
            "seed": seed,
            "split": args.split,
            "checkpoint_best_epoch": state["best_epoch"],
            "cross_entropy": cross_entropy,
            "inference_seconds": inference_seconds,
            "mean_inference_ms_per_utterance": inference_seconds * 1000 / len(frame),
            "device": device.type,
            "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
    )
    (result_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    checkpoint_files = {
        filename: sha256(checkpoint / filename)
        for filename in ["config.json", "model.safetensors", "training_state.json"]
        if (checkpoint / filename).exists()
    }
    metadata = {
        "command": " ".join(sys.argv),
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "checkpoint_files_sha256": checkpoint_files,
        "evaluation_csv_sha256": sha256(data_path),
        "test_used_for_model_selection": False,
        "test_evaluation_explicitly_approved": args.split == "test" and args.allow_test,
    }
    (result_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Results: {result_dir}")


if __name__ == "__main__":
    main()
