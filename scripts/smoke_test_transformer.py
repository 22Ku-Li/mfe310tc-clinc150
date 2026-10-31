"""Run an offline tiny-model smoke test of M1/M2 training and evaluation.

This test uses random weights and synthetic utterances. Its metrics are not
coursework evidence and all generated result/checkpoint directories are removed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import yaml

from src.config import load_config
from src.data import PROJECT_ROOT


def build_tiny_base(base_dir: Path, vocab: dict[str, int]) -> None:
    """Create a local DistilBERT and WordPiece tokenizer without network access."""
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
    from transformers import DistilBertConfig, DistilBertModel, PreTrainedTokenizerFast

    backend = Tokenizer(models.WordPiece(vocab=vocab, unk_token="[UNK]"))
    backend.normalizer = normalizers.BertNormalizer(lowercase=True)
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])],
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        model_max_length=12,
    )
    config = DistilBertConfig(
        vocab_size=len(vocab),
        max_position_embeddings=32,
        dim=24,
        hidden_dim=48,
        n_layers=1,
        n_heads=2,
        dropout=0.1,
        attention_dropout=0.1,
        seq_classif_dropout=0.1,
    )
    model = DistilBertModel(config)
    model.save_pretrained(base_dir, safe_serialization=True)
    tokenizer.save_pretrained(base_dir)


def synthetic_frames(data_dir: Path) -> None:
    """Create tiny train/val/test files with the required 151-label contract."""
    labels = [f"intent_{index:03d}" for index in range(150)] + ["oos"]
    train_rows = []
    for label_index, label in enumerate(labels):
        for variant in range(2):
            train_rows.append(
                {
                    "sample_id": f"train_{label_index}_{variant}",
                    "text": f"intent example number {label_index % 10} variant {variant}",
                    "label": label,
                }
            )
    evaluation_rows = [
        {
            "sample_id": f"evaluation_{index}",
            "text": f"intent example number {index % 10} validation",
            "label": label,
        }
        for index, label in enumerate(labels)
    ]
    pd.DataFrame(train_rows).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame(evaluation_rows).to_csv(data_dir / "val.csv", index=False)
    pd.DataFrame(evaluation_rows).to_csv(data_dir / "test.csv", index=False)


def run_command(command: list[str], expected_returncode: int = 0) -> subprocess.CompletedProcess:
    """Run one smoke step and preserve full logs on failure."""
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    if completed.returncode != expected_returncode:
        print(completed.stdout)
        print(completed.stderr, file=sys.stderr)
        raise RuntimeError(
            f"Command returned {completed.returncode}, expected {expected_returncode}: {' '.join(command)}"
        )
    return completed


def main() -> None:
    artifacts = PROJECT_ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    experiments = ["smoke_m1", "smoke_m2"]
    for experiment in experiments:
        shutil.rmtree(PROJECT_ROOT / "results" / experiment, ignore_errors=True)
        shutil.rmtree(PROJECT_ROOT / "checkpoints" / experiment, ignore_errors=True)

    try:
        with tempfile.TemporaryDirectory(prefix="transformer_smoke_", dir=artifacts) as temporary:
            root = Path(temporary)
            base_dir = root / "tiny_base"
            data_dir = root / "data"
            base_dir.mkdir()
            data_dir.mkdir()
            words = ["intent", "example", "number", "variant", "validation", "0", "1"]
            vocab = {
                token: index
                for index, token in enumerate(
                    ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + words
                )
            }
            build_tiny_base(base_dir, vocab)
            synthetic_frames(data_dir)

            for experiment, source_config in zip(
                experiments, ["configs/distilbert.yaml", "configs/rdrop.yaml"]
            ):
                config = load_config(PROJECT_ROOT / source_config)
                config["experiment_name"] = experiment
                config["seed"] = 7
                config["model"]["local_model_path"] = str(base_dir)
                config["model"]["max_length"] = 12
                config["training"].update(
                    {
                        "device": "cpu",
                        "epochs": 1,
                        "batch_size": 32,
                        "mixed_precision": False,
                        "num_workers": 0,
                    }
                )
                for split in ["train", "val", "test"]:
                    config["data"][f"{split}_path"] = str(data_dir / f"{split}.csv")
                config_path = root / f"{experiment}.yaml"
                config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

                run_command(
                    [
                        sys.executable,
                        "-m",
                        "src.train_transformer",
                        "--config",
                        str(config_path),
                        "--device",
                        "cpu",
                    ]
                )
                run_dir = PROJECT_ROOT / "results" / experiment / "seed_7"
                history = pd.read_csv(run_dir / "history.csv")
                predictions = pd.read_csv(run_dir / "val/predictions.csv")
                if len(history) != 1 or len(predictions) != 151:
                    raise AssertionError(f"Unexpected smoke output shape for {experiment}")
                consistency = history.iloc[0]["train_consistency_loss"]
                if experiment == "smoke_m1" and consistency != 0.0:
                    raise AssertionError("M1 smoke test unexpectedly used a consistency loss")
                if experiment == "smoke_m2" and consistency <= 0.0:
                    raise AssertionError("M2 smoke test did not activate R-Drop consistency")

                checkpoint = PROJECT_ROOT / "checkpoints" / experiment / "seed_7/best"
                run_command(
                    [
                        sys.executable,
                        "-m",
                        "src.evaluate_transformer",
                        "--checkpoint",
                        str(checkpoint),
                        "--split",
                        "val",
                        "--device",
                        "cpu",
                    ]
                )
                run_command(
                    [
                        sys.executable,
                        "-m",
                        "src.evaluate_transformer",
                        "--checkpoint",
                        str(checkpoint),
                        "--split",
                        "test",
                        "--device",
                        "cpu",
                    ],
                    expected_returncode=2,
                )

        print(
            json.dumps(
                {
                    "status": "success",
                    "m1_training_and_reload": "passed",
                    "m2_rdrop_training_and_reload": "passed",
                    "test_guard": "passed",
                    "metrics_used_as_coursework_evidence": False,
                },
                indent=2,
            )
        )
    finally:
        for experiment in experiments:
            shutil.rmtree(PROJECT_ROOT / "results" / experiment, ignore_errors=True)
            shutil.rmtree(PROJECT_ROOT / "checkpoints" / experiment, ignore_errors=True)


if __name__ == "__main__":
    main()
