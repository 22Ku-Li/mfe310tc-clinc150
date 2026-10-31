"""Evaluate the frozen M0 baseline on deterministic spelling-noise challenges."""

from __future__ import annotations

import argparse
import json
import random
import string
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data import PROJECT_ROOT
from src.evaluate import compute_classification_metrics

NOISE_SEED = 20261028


def perturb_text(text: str, rate: float, seed: int) -> str:
    """Apply reproducible deletion, adjacent swap, and substitution edits.

    Only alphabetic characters are targeted. At least one edit is made so every
    non-empty utterance participates in the challenge. The same noisy inputs
    can be reused for every model, ensuring a fair robustness comparison.
    """
    if not 0.0 < rate <= 1.0:
        raise ValueError("Noise rate must be in (0, 1]")
    rng = random.Random(seed)
    characters = list(text)
    candidate_positions = [index for index, character in enumerate(characters) if character.isalpha()]
    if not candidate_positions:
        return text
    edit_count = max(1, round(len(candidate_positions) * rate))
    operations = ("delete", "swap", "substitute")
    for edit_index in range(edit_count):
        positions = [index for index, character in enumerate(characters) if character.isalpha()]
        if not positions:
            break
        operation = operations[(edit_index + rng.randrange(len(operations))) % len(operations)]
        position = rng.choice(positions)
        if operation == "delete":
            characters.pop(position)
        elif operation == "swap" and position + 1 < len(characters) and characters[position + 1].isalpha():
            characters[position], characters[position + 1] = characters[position + 1], characters[position]
        else:
            original = characters[position]
            replacement = rng.choice(string.ascii_lowercase)
            if original.isupper():
                replacement = replacement.upper()
            characters[position] = replacement
    return "".join(characters)


def evaluate_pipeline(pipeline, frame: pd.DataFrame, text_column: str, labels: list[str]) -> tuple[dict, pd.DataFrame]:
    """Evaluate one text variant and return metrics plus row-level predictions."""
    predictions = pipeline.predict(frame[text_column])
    probabilities = pipeline.predict_proba(frame[text_column])
    classes = pipeline.named_steps["classifier"].classes_.tolist()
    oos_index = classes.index("oos")
    oos_scores = probabilities[:, oos_index]
    metrics = compute_classification_metrics(
        frame["label"], predictions, labels=labels, oos_label="oos", oos_scores=oos_scores
    )
    predicted_ids = probabilities.argmax(axis=1)
    output = frame[["sample_id", "text", "label"]].copy()
    output["noisy_text"] = frame[text_column]
    output = output.rename(columns={"label": "true_label"})
    output["predicted_label"] = predictions
    output["predicted_probability"] = probabilities[np.arange(len(predicted_ids)), predicted_ids]
    output["oos_score"] = oos_scores
    output["correct"] = output["true_label"] == output["predicted_label"]
    return metrics, output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--allow-test", action="store_true")
    args = parser.parse_args()
    if args.split == "test" and not args.allow_test:
        parser.error("Noise evaluation on test requires --allow-test after final configuration freeze")

    frame = pd.read_csv(PROJECT_ROOT / f"data/processed/{args.split}.csv")
    pipeline = joblib.load(PROJECT_ROOT / "artifacts/m0_tfidf_logreg.joblib")
    labels = sorted(frame["label"].unique().tolist())
    output_dir = PROJECT_ROOT / "results/robustness/m0_tfidf_logreg" / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    all_predictions = []
    for rate in [0.0, 0.05, 0.10]:
        if rate == 0.0:
            frame["noisy_text"] = frame["text"]
        else:
            frame["noisy_text"] = [
                perturb_text(text, rate, NOISE_SEED + index)
                for index, text in enumerate(frame["text"])
            ]
        metrics, predictions = evaluate_pipeline(pipeline, frame, "noisy_text", labels)
        metrics["noise_rate"] = rate
        metrics["seed_policy"] = f"per-row seed = {NOISE_SEED} + row index"
        rows.append(metrics)
        predictions["noise_rate"] = rate
        all_predictions.append(predictions)

    metrics_frame = pd.DataFrame(rows)
    for metric in ["accuracy", "macro_f1", "in_scope_accuracy", "in_scope_macro_f1", "oos_f1", "oos_auroc"]:
        clean_value = float(metrics_frame.loc[metrics_frame["noise_rate"] == 0.0, metric].iloc[0])
        metrics_frame[f"{metric}_delta_from_clean"] = metrics_frame[metric] - clean_value
    metrics_frame.to_csv(output_dir / "noise_metrics.csv", index=False)
    pd.concat(all_predictions, ignore_index=True).to_csv(output_dir / "noise_predictions.csv", index=False)
    (output_dir / "noise_protocol.json").write_text(
        json.dumps(
            {
                "noise_seed": NOISE_SEED,
                "rates": [0.0, 0.05, 0.10],
                "operations": ["delete", "swap", "substitute"],
                "split": args.split,
                "same_inputs_for_all_models": True,
                "test_used_for_model_selection": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(metrics_frame[["noise_rate", "accuracy", "macro_f1", "oos_f1", "oos_auroc"]].to_string(index=False))
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
