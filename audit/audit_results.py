"""Independently recompute metrics from saved predictions and audit summaries.

This script intentionally does not import the production evaluation helpers.
It checks row-level evidence, split identity, metric definitions, seed summaries,
paired deltas, and final report-analysis aggregates.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
METRICS = ["accuracy", "macro_f1", "in_scope_accuracy", "in_scope_macro_f1", "oos_precision", "oos_recall", "oos_f1", "oos_auroc"]
TOLERANCE = 1e-9


def recompute(frame: pd.DataFrame) -> dict[str, float]:
    """Reproduce the metric definitions directly from prediction columns."""
    true = frame["true_label"].to_numpy()
    pred = frame["predicted_label"].to_numpy()
    labels = sorted(set(true) | set(pred))
    in_scope = true != "oos"
    in_scope_labels = [label for label in labels if label != "oos"]
    oos_precision, oos_recall, oos_f1, _ = precision_recall_fscore_support(
        true == "oos", pred == "oos", average="binary", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, labels=labels, average="macro", zero_division=0)),
        "in_scope_accuracy": float(accuracy_score(true[in_scope], pred[in_scope])),
        "in_scope_macro_f1": float(f1_score(true[in_scope], pred[in_scope], labels=in_scope_labels, average="macro", zero_division=0)),
        "oos_precision": float(oos_precision),
        "oos_recall": float(oos_recall),
        "oos_f1": float(oos_f1),
        "oos_auroc": float(roc_auc_score(true == "oos", frame["oos_score"].to_numpy())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit stored metrics; checkpoint provenance is checked only when available.")
    parser.add_argument("--output-dir", type=Path, default=RESULTS / "audit" / "current_recheck", help="Separate output directory; historical audits are never overwritten")
    parser.add_argument("--require-checkpoints", action="store_true", help="Fail when any of the six model checkpoints is missing")
    args = parser.parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    checks = 0

    def check(condition: bool, message: str) -> None:
        nonlocal checks
        checks += 1
        if not condition:
            errors.append(message)

    prediction_paths = sorted(RESULTS.glob("m0_tfidf_logreg/*/predictions.csv"))
    prediction_paths += sorted(RESULTS.glob("m1_textcnn_*/seed_*/*/predictions.csv"))
    prediction_paths += sorted(RESULTS.glob("m2_textcnn_*/seed_*/*/predictions.csv"))
    prediction_paths += sorted((RESULTS / "robustness").glob("m1_textcnn_*/seed_*/noise_*/*/predictions.csv"))
    prediction_paths += sorted((RESULTS / "robustness").glob("m2_textcnn_*/seed_*/noise_*/*/predictions.csv"))
    check(len(prediction_paths) == 30, f"Expected 30 TextCNN/M0 prediction files, found {len(prediction_paths)}")

    audit_rows = []
    metric_lookup: dict[str, dict] = {}
    for predictions_path in prediction_paths:
        relative = predictions_path.relative_to(RESULTS)
        metrics_path = predictions_path.parent / "metrics.json"
        check(metrics_path.exists(), f"Missing metrics file for {relative}")
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        frame = pd.read_csv(predictions_path)
        key = str(relative).replace("\\", "/")
        metric_lookup[key] = metrics
        split = metrics["split"]
        original = pd.read_csv(PROJECT_ROOT / f"data/processed/{split}.csv")
        check(len(frame) == len(original), f"Row count mismatch in {relative}")
        check(frame["sample_id"].is_unique, f"Duplicate sample IDs in {relative}")
        check(frame["sample_id"].tolist() == original["sample_id"].tolist(), f"Split/order mismatch in {relative}")
        check(frame["text"].tolist() == original["text"].tolist(), f"Original text mismatch in {relative}")
        check(frame["true_label"].tolist() == original["label"].tolist(), f"Original label mismatch in {relative}")
        check(frame["correct"].eq(frame["true_label"] == frame["predicted_label"]).all(), f"Correct flags invalid in {relative}")
        for column in ["predicted_probability", "true_label_probability", "oos_score"]:
            check(frame[column].between(0, 1).all(), f"Probability outside [0,1] in {relative}: {column}")
        oos_predicted = frame["predicted_label"] == "oos"
        check(np.allclose(frame.loc[oos_predicted, "predicted_probability"], frame.loc[oos_predicted, "oos_score"]), f"OOS predicted probability mismatch in {relative}")
        oos_true = frame["true_label"] == "oos"
        check(np.allclose(frame.loc[oos_true, "true_label_probability"], frame.loc[oos_true, "oos_score"]), f"OOS true probability mismatch in {relative}")

        recomputed = recompute(frame)
        max_delta = 0.0
        for metric in METRICS:
            delta = abs(recomputed[metric] - float(metrics[metric]))
            max_delta = max(max_delta, delta)
            check(delta <= TOLERANCE, f"Metric mismatch {metric} in {relative}: saved={metrics[metric]} recomputed={recomputed[metric]}")
        audit_rows.append({"file": key, "rows": len(frame), "max_metric_delta": max_delta, "status": "pass" if max_delta <= TOLERANCE else "fail"})

    # M0 combined noise predictions require a separate grouped audit.
    for split in ["val", "test"]:
        noise_dir = RESULTS / f"robustness/m0_tfidf_logreg/{split}"
        frame = pd.read_csv(noise_dir / "noise_predictions.csv")
        saved = pd.read_csv(noise_dir / "noise_metrics.csv")
        for rate, group in frame.groupby("noise_rate"):
            row = saved.loc[saved["noise_rate"] == rate].iloc[0]
            recomputed = recompute(group)
            for metric in METRICS:
                check(abs(recomputed[metric] - row[metric]) <= TOLERANCE, f"M0 noise mismatch {metric} {split}/{rate}")

    # Verify that every model sees byte-identical noisy utterances.
    for split in ["val", "test"]:
        baseline_noise = pd.read_csv(RESULTS / f"robustness/m0_tfidf_logreg/{split}/noise_predictions.csv")
        for rate in [0.05, 0.10]:
            reference = baseline_noise.loc[baseline_noise["noise_rate"] == rate]
            reference_ids = reference["sample_id"].tolist()
            reference_text = reference["noisy_text"].tolist()
            for experiment in ["m1_textcnn_ce", "m2_textcnn_rdrop"]:
                for seed in [13, 42, 87]:
                    path = RESULTS / f"robustness/{experiment}/seed_{seed}/noise_{rate:.2f}/{split}/predictions.csv"
                    if not path.exists():
                        continue
                    candidate = pd.read_csv(path)
                    check(candidate["sample_id"].tolist() == reference_ids, f"Noise sample order mismatch {experiment}/{seed}/{split}/{rate}")
                    check(candidate["model_input_text"].tolist() == reference_text, f"Noise input mismatch {experiment}/{seed}/{split}/{rate}")

    # Recalculate all clean seed summaries.
    seed_summary = pd.read_csv(RESULTS / "summary/seed_summary.csv")
    for row in seed_summary.itertuples():
        for seed in [13, 42, 87]:
            path = RESULTS / f"{row.experiment}/seed_{seed}/{row.split}/metrics.json"
            check(path.exists(), f"Missing clean seed metrics: {path}")
            if not path.exists():
                continue
            metrics = json.loads(path.read_text(encoding="utf-8"))
            for metric in METRICS:
                values = [json.loads((RESULTS / f"{row.experiment}/seed_{candidate}/{row.split}/metrics.json").read_text(encoding="utf-8"))[metric] for candidate in [13, 42, 87]]
                check(abs(getattr(row, f"{metric}_mean") - float(np.mean(values))) <= TOLERANCE, f"Seed mean mismatch {row.experiment}/{row.split}/{metric}")
                check(abs(getattr(row, f"{metric}_std") - float(np.std(values, ddof=1))) <= TOLERANCE, f"Seed SD mismatch {row.experiment}/{row.split}/{metric}")

    # Recalculate robustness summaries.
    robustness_summary = pd.read_csv(RESULTS / "summary/robustness_seed_summary.csv")
    for row in robustness_summary.itertuples():
        expected_count = int(row.accuracy_count)
        values_by_metric = {metric: [] for metric in METRICS}
        for seed in [13, 42, 87]:
            path = RESULTS / f"robustness/{row.experiment}/seed_{seed}/noise_{row.noise_rate:.2f}/{row.split}/metrics.json"
            if not path.exists():
                check(expected_count < 3, f"Missing noise seed metrics: {path}")
                continue
            metrics = json.loads(path.read_text(encoding="utf-8"))
            for metric in METRICS:
                values_by_metric[metric].append(metrics[metric])
        for metric in METRICS:
            values = values_by_metric[metric]
            check(len(values) == expected_count, f"Noise seed count mismatch {row.experiment}/{row.split}/{row.noise_rate}/{metric}")
            if len(values) == expected_count:
                check(abs(getattr(row, f"{metric}_mean") - float(np.mean(values))) <= TOLERANCE, f"Noise mean mismatch {row.experiment}/{row.split}/{row.noise_rate}/{metric}")
                actual_std = getattr(row, f"{metric}_std")
                if len(values) == 1:
                    check(pd.isna(actual_std), f"Single-seed noise SD should be blank {row.experiment}/{row.split}/{row.noise_rate}/{metric}")
                else:
                    check(abs(actual_std - float(np.std(values, ddof=1))) <= TOLERANCE, f"Noise SD mismatch {row.experiment}/{row.split}/{row.noise_rate}/{metric}")

    # Check every clean paired delta against its source metrics.
    clean_pairs = pd.read_csv(RESULTS / "summary/textcnn_rdrop_minus_ce_paired.csv")
    for row in clean_pairs.itertuples():
        m1 = json.loads((RESULTS / f"m1_textcnn_ce/seed_{row.seed}/{row.split}/metrics.json").read_text(encoding="utf-8"))
        m2 = json.loads((RESULTS / f"m2_textcnn_rdrop/seed_{row.seed}/{row.split}/metrics.json").read_text(encoding="utf-8"))
        for metric in METRICS:
            check(abs(getattr(row, f"{metric}_delta") - (m2[metric] - m1[metric])) <= TOLERANCE, f"Clean paired delta mismatch {row.split}/{row.seed}/{metric}")

    # Check every noise paired delta against its source metrics.
    noise_pairs = pd.read_csv(RESULTS / "summary/textcnn_robustness_paired.csv")
    for row in noise_pairs.itertuples():
        base = RESULTS / "robustness"
        m1_path = base / f"m1_textcnn_ce/seed_{row.seed}/noise_{row.noise_rate:.2f}/{row.split}/metrics.json"
        m2_path = base / f"m2_textcnn_rdrop/seed_{row.seed}/noise_{row.noise_rate:.2f}/{row.split}/metrics.json"
        m1 = json.loads(m1_path.read_text(encoding="utf-8"))
        m2 = json.loads(m2_path.read_text(encoding="utf-8"))
        for metric in METRICS:
            check(abs(getattr(row, f"{metric}_delta") - (m2[metric] - m1[metric])) <= TOLERANCE, f"Noise paired delta mismatch {row.split}/{row.seed}/{row.noise_rate}/{metric}")

    # Check the aggregate consumed by the report builder.
    analysis = json.loads((RESULTS / "report_assets/final_analysis_summary.json").read_text(encoding="utf-8"))
    test_summary = seed_summary.loc[seed_summary["split"] == "test"].set_index("experiment")
    for experiment, prefix in [("m1_textcnn_ce", "m1"), ("m2_textcnn_rdrop", "m2")]:
        for metric in analysis[f"{prefix}_test_mean"]:
            check(abs(analysis[f"{prefix}_test_mean"][metric] - test_summary.loc[experiment, f"{metric}_mean"]) <= TOLERANCE, f"Report aggregate mean mismatch {prefix}/{metric}")
            check(abs(analysis[f"{prefix}_test_std"][metric] - test_summary.loc[experiment, f"{metric}_std"]) <= TOLERANCE, f"Report aggregate SD mismatch {prefix}/{metric}")
    for row in clean_pairs.loc[clean_pairs["split"] == "test"].itertuples():
        check(abs(analysis["clean_macro_f1_delta_by_seed"][str(row.seed)] - row.macro_f1_delta) <= TOLERANCE, f"Report clean delta mismatch seed {row.seed}")
    for experiment in ["m1_textcnn_ce", "m2_textcnn_rdrop"]:
        for rate in [0.05, 0.10]:
            expected = robustness_summary.loc[(robustness_summary["split"] == "test") & (robustness_summary["experiment"] == experiment) & (robustness_summary["noise_rate"] == rate), "macro_f1_mean"].iloc[0]
            check(abs(analysis["noise_macro_f1_mean"][experiment][str(rate)] - expected) <= TOLERANCE, f"Report noise aggregate mismatch {experiment}/{rate}")

    # Independently audit the raw-data counts and overlap claims.
    raw = json.loads((PROJECT_ROOT / "data/raw/data_full.json").read_text(encoding="utf-8"))
    split_sources = {"train": ("train", "oos_train"), "val": ("val", "oos_val"), "test": ("test", "oos_test")}
    frames = {}
    for split, keys in split_sources.items():
        records = []
        for key in keys:
            records.extend(raw[key])
        frames[split] = pd.DataFrame(records, columns=["text", "label"])
        frames[split]["normalized"] = frames[split]["text"].str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
    expected_counts = {"train": 15100, "val": 3100, "test": 5500}
    for split, frame in frames.items():
        check(len(frame) == expected_counts[split], f"Raw split count mismatch {split}")
        check(not frame[["text", "label"]].isna().any().any(), f"Raw missing values in {split}")
        check(not frame["text"].duplicated().any(), f"Raw within-split duplicate text in {split}")
    quality = json.loads((RESULTS / "data_checks/data_quality_summary.json").read_text(encoding="utf-8"))
    for first, second in [("train", "val"), ("train", "test"), ("val", "test")]:
        exact = len(set(frames[first]["text"]) & set(frames[second]["text"]))
        normalized = len(set(frames[first]["normalized"]) & set(frames[second]["normalized"]))
        saved_exact = next(row for row in quality["cross_split_overlaps"] if row["split_a"] == first and row["split_b"] == second and row["kind"] == "exact")
        saved_normalized = next(row for row in quality["cross_split_overlaps"] if row["split_a"] == first and row["split_b"] == second and row["kind"] == "normalized")
        check(exact == saved_exact["shared_unique_texts"], f"Raw exact overlap mismatch {first}/{second}")
        check(normalized == saved_normalized["shared_unique_texts"], f"Raw normalized overlap mismatch {first}/{second}")
    all_rows = pd.concat(frames.values(), ignore_index=True)
    conflicts = int((all_rows.groupby("normalized")["label"].nunique() > 1).sum())
    check(conflicts == quality["normalized_texts_with_conflicting_labels"], "Raw conflicting-label count mismatch")

    # Flags are saved assertions, not independent proof of historical ordering.
    # Filesystem timestamps are not reliable provenance after copying or archiving.
    checkpoints_available = 0
    checkpoints_missing = []
    for experiment in ["m1_textcnn_ce", "m2_textcnn_rdrop"]:
        for seed in [13, 42, 87]:
            run_dir = RESULTS / f"{experiment}/seed_{seed}"
            metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
            check(metadata["test_used_for_model_selection"] is False, f"Test selection flag invalid {experiment}/{seed}")
            check(metadata["approved_remote_linux_gpu_used"] is False, f"GPU claim invalid {experiment}/{seed}")
            checkpoint = PROJECT_ROOT / f"checkpoints/{experiment}/seed_{seed}/best"
            if not ((checkpoint / "training_state.json").exists() and (checkpoint / "model.pt").exists()):
                checkpoints_missing.append(f"{experiment}/seed_{seed}")
                if args.require_checkpoints:
                    check(False, f"Missing required checkpoint {experiment}/{seed}")
                continue
            checkpoints_available += 1
            state = json.loads((checkpoint / "training_state.json").read_text(encoding="utf-8"))
            validation_metrics = json.loads((run_dir / "val/metrics.json").read_text(encoding="utf-8"))
            check(state["best_epoch"] == validation_metrics["best_epoch"], f"Best epoch mismatch {experiment}/{seed}")
            check(state["seed"] == seed, f"Checkpoint seed mismatch {experiment}/{seed}")
            check(state["config"]["seed"] == seed, f"Config snapshot seed mismatch {experiment}/{seed}")

    summary = {
        "status": ("failed" if errors else "passed_with_missing_checkpoints" if checkpoints_missing else "passed"),
        "checks_performed": checks,
        "prediction_files_audited": len(prediction_paths),
        "m0_noise_groups_audited": 6,
        "raw_rows_audited": sum(len(frame) for frame in frames.values()),
        "checkpoints_available": checkpoints_available,
        "checkpoints_missing": checkpoints_missing,
        "provenance_limit": "Saved metadata assertions and metrics cannot independently prove historical training or test ordering.",
        "errors": errors,
    }
    (output_dir / "independent_metric_audit.csv").write_text(pd.DataFrame(audit_rows).to_csv(index=False), encoding="utf-8")
    (output_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
