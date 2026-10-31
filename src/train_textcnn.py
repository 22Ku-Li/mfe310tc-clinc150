"""Train M1 TextCNN or M2 TextCNN + R-Drop without external model downloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import platform
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.config import load_config, resolve_project_path
from src.data import PROJECT_ROOT, sha256
from src.evaluate import compute_classification_metrics
from src.losses import training_objective
from src.textcnn_common import TextCNNClassifier, TextIntentDataset, Vocabulary
from src.transformer_common import (
    load_and_validate_split,
    make_data_loader,
    move_inputs_to_device,
    predict_probabilities,
    resolve_device,
    save_evaluation_outputs,
    set_reproducible_seed,
)


def validate_config(config: dict) -> None:
    loss_name = config["model"]["loss"]
    alpha = float(config["rdrop"]["alpha"])
    if loss_name == "cross_entropy" and alpha != 0.0:
        raise ValueError("M1 cross_entropy requires rdrop.alpha=0")
    if loss_name == "rdrop" and alpha <= 0.0:
        raise ValueError("M2 rdrop requires a positive rdrop.alpha")
    if loss_name not in {"cross_entropy", "rdrop"}:
        raise ValueError(f"Unsupported loss: {loss_name}")


def configure_logger(log_path) -> logging.Logger:
    logger = logging.getLogger("textcnn_training")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in [logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")]:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.propagate = False
    return logger


def save_checkpoint(model, vocabulary, checkpoint_dir, config, labels, epoch, macro_f1):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_dir / "model.pt")
    vocabulary.save(checkpoint_dir / "vocabulary.json")
    state = {
        "experiment_name": config["experiment_name"],
        "seed": config["seed"],
        "best_epoch": epoch,
        "best_validation_macro_f1": macro_f1,
        "labels": labels,
        "config": config,
        "approved_remote_linux_gpu_used": False,
    }
    (checkpoint_dir / "training_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default=None)
    args = parser.parse_args()

    config_path = resolve_project_path(args.config)
    config = load_config(config_path)
    if args.seed is not None:
        config["seed"] = args.seed
    if args.device is not None:
        config["training"]["device"] = args.device
    validate_config(config)
    seed = int(config["seed"])
    training = config["training"]
    set_reproducible_seed(seed, training["deterministic"])
    device = resolve_device(training["device"])
    if device.type != "cpu":
        logger_device_note = "CUDA is optional for this script; the assessed runs record the actual device."
    else:
        logger_device_note = "CPU-only run; approved remote GPU practice is not claimed."

    experiment = config["experiment_name"]
    run_dir = PROJECT_ROOT / "results" / experiment / f"seed_{seed}"
    validation_dir = run_dir / "val"
    checkpoint_dir = PROJECT_ROOT / "checkpoints" / experiment / f"seed_{seed}" / "best"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logger(run_dir / "training.log")
    (run_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    train_path = resolve_project_path(config["data"]["train_path"])
    validation_path = resolve_project_path(config["data"]["val_path"])
    train = load_and_validate_split(train_path)
    validation = load_and_validate_split(validation_path)
    if set(train["sample_id"]) & set(validation["sample_id"]):
        raise ValueError("Train/validation sample ID leakage detected")
    labels = sorted(train["label"].unique().tolist())
    if len(labels) != 151 or config["evaluation"]["oos_label"] not in labels:
        raise ValueError("Expected 150 intents plus OOS")
    label_to_id = {label: index for index, label in enumerate(labels)}

    vocabulary = Vocabulary.build(
        train["text"], config["model"]["vocab_size"], config["model"]["min_token_frequency"]
    )
    model = TextCNNClassifier(
        len(vocabulary),
        len(labels),
        config["model"]["embedding_dim"],
        config["model"]["filter_sizes"],
        config["model"]["num_filters"],
        config["model"]["dropout"],
    ).to(device)
    train_dataset = TextIntentDataset(train, vocabulary, label_to_id, config["model"]["max_length"])
    validation_dataset = TextIntentDataset(validation, vocabulary, label_to_id, config["model"]["max_length"])
    train_loader = make_data_loader(train_dataset, training["batch_size"], True, seed, training["num_workers"], False)
    validation_loader = make_data_loader(validation_dataset, training["batch_size"], False, seed, training["num_workers"], False)

    optimizer = AdamW(model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"])
    accumulation = int(training["gradient_accumulation_steps"])
    total_updates = math.ceil(len(train_loader) / accumulation) * int(training["epochs"])
    warmup_steps = int(total_updates * training["warmup_ratio"])

    def learning_rate_factor(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, (total_updates - step) / max(1, total_updates - warmup_steps))

    scheduler = LambdaLR(optimizer, learning_rate_factor)
    logger.info("Experiment=%s seed=%d device=%s", experiment, seed, device)
    logger.info(logger_device_note)
    logger.info("Train=%d validation=%d labels=%d vocabulary=%d parameters=%d", len(train), len(validation), len(labels), len(vocabulary), sum(parameter.numel() for parameter in model.parameters()))
    logger.info("Loss=%s R-Drop alpha=%s", config["model"]["loss"], config["rdrop"]["alpha"])

    best_macro_f1 = -float("inf")
    best_epoch = 0
    best_metrics = None
    stale_epochs = 0
    history = []
    training_start = time.perf_counter()

    for epoch in range(1, int(training["epochs"]) + 1):
        epoch_start = time.perf_counter()
        model.train()
        totals = {"loss": 0.0, "classification": 0.0, "consistency": 0.0, "rows": 0}
        optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(train_loader):
            inputs, batch_labels = move_inputs_to_device(batch, device)
            first_logits = model(**inputs).logits
            second_logits = model(**inputs).logits if config["model"]["loss"] == "rdrop" else None
            total_loss, classification_loss, consistency_loss = training_objective(
                first_logits,
                batch_labels,
                second_logits=second_logits,
                rdrop_alpha=float(config["rdrop"]["alpha"]),
            )
            (total_loss / accumulation).backward()
            rows = len(batch_labels)
            totals["loss"] += total_loss.detach().item() * rows
            totals["classification"] += classification_loss.item() * rows
            totals["consistency"] += consistency_loss.item() * rows
            totals["rows"] += rows
            if (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), training["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        true_ids, probabilities, validation_ce, inference_seconds = predict_probabilities(
            model, validation_loader, device, False
        )
        predicted_labels = [labels[int(index)] for index in probabilities.argmax(axis=1)]
        epoch_metrics = compute_classification_metrics(
            validation["label"],
            predicted_labels,
            labels,
            config["evaluation"]["oos_label"],
            probabilities[:, labels.index(config["evaluation"]["oos_label"])],
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
        history.append({
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
        })
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        logger.info("Epoch %d | train %.5f | val CE %.5f | Macro-F1 %.5f | OOS-F1 %.5f", epoch, history[-1]["train_total_loss"], validation_ce, epoch_metrics["macro_f1"], epoch_metrics["oos_f1"])

        if epoch_metrics["macro_f1"] - best_macro_f1 > float(training["early_stopping_min_delta"]):
            best_macro_f1 = float(epoch_metrics["macro_f1"])
            best_epoch = epoch
            stale_epochs = 0
            best_metrics = save_evaluation_outputs(
                validation,
                labels,
                true_ids,
                probabilities,
                validation_dir,
                config["evaluation"]["oos_label"],
                extra_metrics=extra_metrics,
            )
            save_checkpoint(model, vocabulary, checkpoint_dir, config, labels, epoch, best_macro_f1)
        else:
            stale_epochs += 1
            if stale_epochs >= int(training["early_stopping_patience"]):
                logger.info("Early stopping after epoch %d", epoch)
                break

    if best_metrics is None:
        raise RuntimeError("No valid checkpoint was produced")
    best_metrics.update({
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "total_training_seconds": time.perf_counter() - training_start,
        "device": device.type,
        "approved_remote_linux_gpu_used": False,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "vocabulary_size": len(vocabulary),
    })
    (validation_dir / "metrics.json").write_text(json.dumps(best_metrics, indent=2), encoding="utf-8")
    metadata = {
        "command": " ".join(sys.argv),
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "torch_version": torch.__version__,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "train_csv_sha256": sha256(train_path),
        "validation_csv_sha256": sha256(validation_path),
        "deterministic_algorithms": training["deterministic"],
        "test_used_for_model_selection": False,
        "approved_remote_linux_gpu_used": False,
        "gpu_practice_requirement_met": False,
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Best epoch=%d Macro-F1=%.6f", best_epoch, best_macro_f1)
    print(json.dumps(best_metrics, indent=2))
    for handler in logger.handlers:
        handler.flush()


if __name__ == "__main__":
    main()
