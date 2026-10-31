"""Train and evaluate the M0 TF-IDF + logistic regression baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from src.data import PROJECT_ROOT, sha256
from src.evaluate import compute_classification_metrics, save_classification_report, save_top_confusions


def load_config(path: Path) -> dict:
    """Load one immutable experiment configuration."""
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_project_path(value: str) -> Path:
    """Resolve a configuration path relative to the repository root."""
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def build_pipeline(config: dict) -> Pipeline:
    """Construct the baseline so preprocessing is fitted on training data only."""
    feature_config = config["features"]
    common = {
        "lowercase": feature_config["lowercase"],
        "strip_accents": feature_config["strip_accents"],
        "sublinear_tf": feature_config["sublinear_tf"],
        "dtype": np.float32,
    }
    word_features = TfidfVectorizer(
        analyzer="word",
        ngram_range=(feature_config["word_ngram_min"], feature_config["word_ngram_max"]),
        min_df=feature_config["word_min_df"],
        max_features=feature_config["word_max_features"],
        **common,
    )
    char_features = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(feature_config["char_ngram_min"], feature_config["char_ngram_max"]),
        min_df=feature_config["char_min_df"],
        max_features=feature_config["char_max_features"],
        **common,
    )
    model_config = config["model"]
    classifier = LogisticRegression(
        solver=model_config["solver"],
        C=model_config["C"],
        max_iter=model_config["max_iter"],
        tol=model_config["tol"],
        random_state=config["seed"],
    )
    return Pipeline(
        [
            ("features", FeatureUnion([("word", word_features), ("char", char_features)])),
            ("classifier", classifier),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/baseline.yaml")
    parser.add_argument("--split", choices=["val", "test"], default=None)
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="Required for test evaluation to prevent accidental test-driven model selection.",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    split = args.split or config["evaluation"]["default_split"]
    if split == "test" and not args.allow_test:
        parser.error("Test evaluation requires --allow-test after all model choices are frozen")

    train_path = resolve_project_path(config["data"]["train_path"])
    evaluation_path = resolve_project_path(config["data"][f"{split}_path"])
    train = pd.read_csv(train_path)
    evaluation = pd.read_csv(evaluation_path)
    if train["sample_id"].duplicated().any() or evaluation["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample IDs detected")
    if set(train["sample_id"]) & set(evaluation["sample_id"]):
        raise ValueError("Train/evaluation sample ID leakage detected")

    labels = sorted(train["label"].unique().tolist())
    if len(labels) != 151 or config["evaluation"]["oos_label"] not in labels:
        raise ValueError("Expected 150 intents plus the OOS class")

    pipeline = build_pipeline(config)
    fit_start = time.perf_counter()
    pipeline.fit(train["text"], train["label"])
    fit_seconds = time.perf_counter() - fit_start

    inference_start = time.perf_counter()
    predictions = pipeline.predict(evaluation["text"])
    probabilities = pipeline.predict_proba(evaluation["text"])
    inference_seconds = time.perf_counter() - inference_start

    classes = pipeline.named_steps["classifier"].classes_.tolist()
    oos_label = config["evaluation"]["oos_label"]
    oos_index = classes.index(oos_label)
    oos_scores = probabilities[:, oos_index]
    predicted_class_indices = np.argmax(probabilities, axis=1)
    predicted_probabilities = probabilities[np.arange(len(evaluation)), predicted_class_indices]
    true_label_to_index = {label: index for index, label in enumerate(classes)}
    true_probabilities = np.asarray(
        [probabilities[row, true_label_to_index[label]] for row, label in enumerate(evaluation["label"])]
    )

    metrics = compute_classification_metrics(
        evaluation["label"], predictions, labels=labels, oos_label=oos_label, oos_scores=oos_scores
    )
    feature_count = int(pipeline.named_steps["features"].transform(evaluation["text"].iloc[:1]).shape[1])
    metrics.update(
        {
            "experiment": config["experiment_name"],
            "split": split,
            "seed": config["seed"],
            "feature_count": feature_count,
            "fit_seconds": fit_seconds,
            "inference_seconds": inference_seconds,
            "mean_inference_ms_per_utterance": inference_seconds * 1000 / len(evaluation),
            "iterations": pipeline.named_steps["classifier"].n_iter_.tolist(),
        }
    )

    result_dir = PROJECT_ROOT / "results" / config["experiment_name"] / split
    result_dir.mkdir(parents=True, exist_ok=True)
    prediction_frame = evaluation[["sample_id", "text", "label"]].copy()
    prediction_frame = prediction_frame.rename(columns={"label": "true_label"})
    prediction_frame["predicted_label"] = predictions
    prediction_frame["predicted_probability"] = predicted_probabilities
    prediction_frame["true_label_probability"] = true_probabilities
    prediction_frame["oos_score"] = oos_scores
    prediction_frame["correct"] = prediction_frame["true_label"] == prediction_frame["predicted_label"]
    prediction_frame.to_csv(result_dir / "predictions.csv", index=False)

    save_classification_report(evaluation["label"], predictions, labels, result_dir / "per_intent_metrics.csv")
    save_top_confusions(evaluation["label"], predictions, labels, result_dir / "top_confusions.csv")
    (result_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (result_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )

    artifact_dir = PROJECT_ROOT / "artifacts"
    artifact_dir.mkdir(exist_ok=True)
    joblib.dump(pipeline, artifact_dir / f"{config['experiment_name']}.joblib")

    metadata = {
        "command": " ".join(sys.argv),
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "train_csv_sha256": sha256(train_path),
        "evaluation_csv_sha256": sha256(evaluation_path),
        "test_used_for_model_selection": False,
    }
    (result_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(f"Results: {result_dir}")


if __name__ == "__main__":
    main()
