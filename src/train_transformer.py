"""Train M1 DistilBERT or M2 DistilBERT + R-Drop on the fixed CLINC150 split."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.optim import AdamW

from src.data import PROJECT_ROOT, sha256
from src.evaluate import compute_classification_metrics
from src.losses import training_objective
from src.transformer_common import (
    TokenizedIntentDataset,
    autocast_context,
    load_and_validate_split,
    load_config,
    make_data_loader,
    move_inputs_to_device,
    predict_probabilities,
    resolve_device,
    resolve_project_path,
    save_evaluation_outputs,
    set_reproducible_seed,
)


def validate_config(config: dict) -> None:
    """Reject configurations that would break the controlled comparison."""
    loss_name = config["model"]["loss"]
    alpha = float(config["rdrop"]["alpha"])
    if loss_name == "cross_entropy" and alpha != 0.0:
        raise ValueError("M1 cross_entropy requires rdrop.alpha=0")
    if loss_name == "rdrop" and alpha <= 0.0:
        raise ValueError("M2 rdrop requires a positive rdrop.alpha")
    if loss_name not in {"cross_entropy", "rdrop"}:
        raise ValueError(f"Unsupported loss: {loss_name}")


def configure_logger(log_path: Path) -> logging.Logger:
    """Write the same training log to the console and a persistent file."""
    logger = logging.getLogger("transformer_training")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None, help="Overrides the YAML seed for multi-seed runs")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    args = parser.parse_args()

    try:
        import transformers
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    except ImportError as error:
        raise SystemExit("Install Transformer dependencies with pip install -r requirements-transformers.txt") from error

    config_path = resolve_project_path(args.config)
    config = load_config(config_path)
    if args.seed is not None:
        config["seed"] = args.seed
    if args.device is not None:
        config["training"]["device"] = args.device
    validate_config(config)

    seed = int(config["seed"])
    training_config = config["training"]
    set_reproducible_seed(seed, deterministic=training_config["deterministic"])
    device = resolve_device(training_config["device"])
    use_amp = bool(training_config["mixed_precision"] and device.type == "cuda")

    experiment = config["experiment_name"]
    run_dir = PROJECT_ROOT / "results" / experiment / f"seed_{seed}"
    validation_dir = run_dir / "val"
    checkpoint_dir = PROJECT_ROOT / "checkpoints" / experiment / f"seed_{seed}" / "best"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logger(run_dir / "training.log")
    (run_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )

    train_path = resolve_project_path(config["data"]["train_path"])
    validation_path = resolve_project_path(config["data"]["val_path"])
    train = load_and_validate_split(train_path)
    validation = load_and_validate_split(validation_path)
    if set(train["sample_id"]) & set(validation["sample_id"]):
        raise ValueError("Train/validation sample ID leakage detected")
    labels = sorted(train["label"].unique().tolist())
    if len(labels) != 151 or config["evaluation"]["oos_label"] not in labels:
        raise ValueError("Expected 150 intents plus the OOS class")
    label_to_id = {label: index for index, label in enumerate(labels)}
    id_to_label = {index: label for label, index in label_to_id.items()}

    local_model_path = resolve_project_path(config["model"]["local_model_path"])
    if not local_model_path.exists():
        raise FileNotFoundError(
            f"Shared base snapshot not found at {local_model_path}. "
            f"Run: python -m src.prepare_model --config {args.config}"
        )
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        local_model_path,
        num_labels=len(labels),
        id2label=id_to_label,
        label2id=label_to_id,
        local_files_only=True,
    )
    model.to(device)

    train_dataset = TokenizedIntentDataset(
        train, tokenizer, label_to_id, config["model"]["max_length"]
    )
    validation_dataset = TokenizedIntentDataset(
        validation, tokenizer, label_to_id, config["model"]["max_length"]
    )
    train_loader = make_data_loader(
        train_dataset,
        training_config["batch_size"],
        True,
        seed,
        training_config["num_workers"],
        device.type == "cuda",
    )
    validation_loader = make_data_loader(
        validation_dataset,
        training_config["batch_size"],
        False,
        seed,
        training_config["num_workers"],
        device.type == "cuda",
    )

    optimizer = AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )
    accumulation = int(training_config["gradient_accumulation_steps"])
    updates_per_epoch = math.ceil(len(train_loader) / accumulation)
    total_updates = updates_per_epoch * int(training_config["epochs"])
    warmup_steps = int(total_updates * training_config["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_updates)
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    logger.info("Experiment=%s seed=%d device=%s AMP=%s", experiment, seed, device, use_amp)
    logger.info("Train rows=%d validation rows=%d labels=%d", len(train), len(validation), len(labels))
    logger.info("Loss=%s R-Drop alpha=%s", config["model"]["loss"], config["rdrop"]["alpha"])
    if device.type == "cuda":
        logger.info("GPU=%s CUDA=%s", torch.cuda.get_device_name(0), torch.version.cuda)

    best_macro_f1 = -float("inf")
    best_epoch = 0
    best_metrics = None
    epochs_without_improvement = 0
    history = []
    training_start = time.perf_counter()

    for epoch in range(1, int(training_config["epochs"]) + 1):
        epoch_start = time.perf_counter()
        model.train()
        totals = {"loss": 0.0, "classification": 0.0, "consistency": 0.0, "rows": 0}
        optimizer.zero_grad(set_to_none=True)

        for batch_index, batch in enumerate(train_loader):
            inputs, batch_labels = move_inputs_to_device(batch, device)
            with autocast_context(device, use_amp):
                first_logits = model(**inputs).logits
                second_logits = None
                if config["model"]["loss"] == "rdrop":
                    second_logits = model(**inputs).logits
                total_loss, classification_loss, consistency_loss = training_objective(
                    first_logits,
                    batch_labels,
                    second_logits=second_logits,
                    rdrop_alpha=float(config["rdrop"]["alpha"]),
                )
                backward_loss = total_loss / accumulation

            if use_amp:
                scaler.scale(backward_loss).backward()
            else:
                backward_loss.backward()

            batch_rows = len(batch_labels)
            totals["loss"] += total_loss.detach().float().item() * batch_rows
            totals["classification"] += classification_loss.float().item() * batch_rows
            totals["consistency"] += consistency_loss.float().item() * batch_rows
            totals["rows"] += batch_rows

            should_update = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(train_loader)
            if should_update:
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), training_config["max_grad_norm"])
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        true_ids, probabilities, validation_ce, inference_seconds = predict_probabilities(
            model, validation_loader, device, use_amp
        )
        predicted_labels = [labels[int(index)] for index in probabilities.argmax(axis=1)]
        oos_scores = probabilities[:, labels.index(config["evaluation"]["oos_label"])]
        epoch_metrics = compute_classification_metrics(
            validation["label"],
            predicted_labels,
            labels=labels,
            oos_label=config["evaluation"]["oos_label"],
            oos_scores=oos_scores,
        )
        extra_metrics = {
            "experiment": experiment,
            "seed": seed,
            "split": "val",
            "epoch": epoch,
            "validation_cross_entropy": validation_ce,
            "inference_seconds": inference_seconds,
            "mean_inference_ms_per_utterance": inference_seconds * 1000 / len(validation),
        }
        epoch_metrics.update(extra_metrics)
        history_row = {
            "epoch": epoch,
            "learning_rate": scheduler.get_last_lr()[0],
            "train_total_loss": totals["loss"] / totals["rows"],
            "train_classification_loss": totals["classification"] / totals["rows"],
            "train_consistency_loss": totals["consistency"] / totals["rows"],
            "validation_cross_entropy": validation_ce,
            "validation_accuracy": epoch_metrics["accuracy"],
            "validation_macro_f1": epoch_metrics["macro_f1"],
            "validation_oos_f1": epoch_metrics["oos_f1"],
            "validation_oos_auroc": epoch_metrics["oos_auroc"],
            "epoch_seconds": time.perf_counter() - epoch_start,
        }
        history.append(history_row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        logger.info(
            "Epoch %d | train loss %.5f | val CE %.5f | val Macro-F1 %.5f | val OOS-F1 %.5f",
            epoch,
            history_row["train_total_loss"],
            validation_ce,
            epoch_metrics["macro_f1"],
            epoch_metrics["oos_f1"],
        )

        improvement = epoch_metrics["macro_f1"] - best_macro_f1
        if improvement > float(training_config["early_stopping_min_delta"]):
            best_macro_f1 = float(epoch_metrics["macro_f1"])
            best_epoch = epoch
            epochs_without_improvement = 0
            best_metrics = save_evaluation_outputs(
                validation,
                labels,
                true_ids,
                probabilities,
                validation_dir,
                config["evaluation"]["oos_label"],
                extra_metrics=extra_metrics,
            )
            model.save_pretrained(checkpoint_dir, safe_serialization=True)
            tokenizer.save_pretrained(checkpoint_dir)
            training_state = {
                "experiment_name": experiment,
                "seed": seed,
                "best_epoch": best_epoch,
                "best_validation_macro_f1": best_macro_f1,
                "labels": labels,
                "config": config,
            }
            (checkpoint_dir / "training_state.json").write_text(
                json.dumps(training_state, indent=2), encoding="utf-8"
            )
            logger.info("Saved new best checkpoint at epoch %d", epoch)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= int(training_config["early_stopping_patience"]):
                logger.info("Early stopping after epoch %d", epoch)
                break

    total_training_seconds = time.perf_counter() - training_start
    if best_metrics is None:
        raise RuntimeError("Training completed without a valid validation checkpoint")
    best_metrics.update(
        {
            "best_epoch": best_epoch,
            "epochs_run": len(history),
            "total_training_seconds": total_training_seconds,
            "device": device.type,
            "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        }
    )
    (validation_dir / "metrics.json").write_text(json.dumps(best_metrics, indent=2), encoding="utf-8")

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
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "train_csv_sha256": sha256(train_path),
        "validation_csv_sha256": sha256(validation_path),
        "base_model_manifest": "results/model_provenance/base_model_manifest.json",
        "deterministic_algorithms": training_config["deterministic"],
        "test_used_for_model_selection": False,
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Best epoch=%d Macro-F1=%.6f", best_epoch, best_macro_f1)
    logger.info("Results: %s", run_dir)
    for handler in logger.handlers:
        handler.flush()


if __name__ == "__main__":
    main()
