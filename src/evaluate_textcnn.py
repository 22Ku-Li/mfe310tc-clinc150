"""Evaluate frozen TextCNN checkpoints on clean or deterministic-noise inputs."""

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
from src.robustness import NOISE_SEED, perturb_text
from src.textcnn_common import TextCNNClassifier, TextIntentDataset, Vocabulary
from src.transformer_common import (
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
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", required=True, choices=["val", "test"])
    parser.add_argument("--noise-rate", type=float, default=0.0, choices=[0.0, 0.05, 0.10])
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--allow-test", action="store_true")
    args = parser.parse_args()
    if args.split == "test" and not args.allow_test:
        parser.error("Test evaluation requires --allow-test after final configuration freeze")

    checkpoint = resolve_project_path(args.checkpoint)
    state = json.loads((checkpoint / "training_state.json").read_text(encoding="utf-8"))
    config = state["config"]
    labels = state["labels"]
    label_to_id = {label: index for index, label in enumerate(labels)}
    seed = int(state["seed"])
    device = resolve_device(args.device)
    set_reproducible_seed(seed, config["training"]["deterministic"])

    original = load_and_validate_split(resolve_project_path(config["data"][f"{args.split}_path"]))
    model_input = original.copy()
    if args.noise_rate > 0:
        model_input["text"] = [
            perturb_text(text, args.noise_rate, NOISE_SEED + index)
            for index, text in enumerate(original["text"])
        ]
    vocabulary = Vocabulary.load(checkpoint / "vocabulary.json")
    model = TextCNNClassifier(
        len(vocabulary),
        len(labels),
        config["model"]["embedding_dim"],
        config["model"]["filter_sizes"],
        config["model"]["num_filters"],
        config["model"]["dropout"],
    ).to(device)
    model.load_state_dict(torch.load(checkpoint / "model.pt", map_location=device, weights_only=True))
    dataset = TextIntentDataset(model_input, vocabulary, label_to_id, config["model"]["max_length"])
    loader = make_data_loader(dataset, config["training"]["batch_size"], False, seed, 0, False)
    true_ids, probabilities, cross_entropy, inference_seconds = predict_probabilities(model, loader, device, False)

    experiment = state["experiment_name"]
    if args.noise_rate > 0:
        result_dir = PROJECT_ROOT / "results/robustness" / experiment / f"seed_{seed}" / f"noise_{args.noise_rate:.2f}" / args.split
    else:
        result_dir = PROJECT_ROOT / "results" / experiment / f"seed_{seed}" / args.split
    metrics = save_evaluation_outputs(
        original,
        labels,
        true_ids,
        probabilities,
        result_dir,
        config["evaluation"]["oos_label"],
        extra_metrics={
            "experiment": experiment,
            "seed": seed,
            "split": args.split,
            "noise_rate": args.noise_rate,
            "checkpoint_best_epoch": state["best_epoch"],
            "cross_entropy": cross_entropy,
            "inference_seconds": inference_seconds,
            "mean_inference_ms_per_utterance": inference_seconds * 1000 / len(original),
            "device": device.type,
            "approved_remote_linux_gpu_used": False,
        },
    )
    predictions_path = result_dir / "predictions.csv"
    predictions = pd.read_csv(predictions_path)
    predictions["model_input_text"] = model_input["text"]
    predictions.to_csv(predictions_path, index=False)
    (result_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    metadata = {
        "command": " ".join(sys.argv),
        "platform": platform.platform(),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "torch_version": torch.__version__,
        "evaluation_csv_sha256": sha256(resolve_project_path(config["data"][f"{args.split}_path"])),
        "noise_seed": NOISE_SEED if args.noise_rate > 0 else None,
        "test_used_for_model_selection": False,
        "approved_remote_linux_gpu_used": False,
    }
    (result_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Results: {result_dir}")


if __name__ == "__main__":
    main()
