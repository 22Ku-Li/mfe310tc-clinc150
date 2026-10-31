"""Aggregate multi-seed metrics and compute paired R-Drop minus CE differences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data import PROJECT_ROOT

METRICS = [
    "accuracy",
    "macro_f1",
    "in_scope_accuracy",
    "in_scope_macro_f1",
    "oos_precision",
    "oos_recall",
    "oos_f1",
    "oos_auroc",
]
EXPECTED_SEEDS = {13, 42, 87}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-all-seeds", action="store_true")
    args = parser.parse_args()

    rows = []
    for metrics_path in sorted((PROJECT_ROOT / "results").glob("m*/seed_*/*/metrics.json")):
        split = metrics_path.parent.name
        if split not in {"val", "test"}:
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        row = {
            "experiment": metrics["experiment"],
            "seed": int(metrics["seed"]),
            "split": split,
            "metrics_path": str(metrics_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        }
        for metric in METRICS:
            row[metric] = metrics.get(metric)
        rows.append(row)
    if not rows:
        raise SystemExit("No multi-seed Transformer metrics found")

    individual = pd.DataFrame(rows).sort_values(["split", "experiment", "seed"])
    output_dir = PROJECT_ROOT / "results/summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    individual.to_csv(output_dir / "individual_runs.csv", index=False)

    aggregated = individual.groupby(["split", "experiment"])[METRICS].agg(["count", "mean", "std"])
    aggregated.columns = [f"{metric}_{statistic}" for metric, statistic in aggregated.columns]
    aggregated = aggregated.reset_index()
    for split, experiment in aggregated[["split", "experiment"]].itertuples(index=False):
        seeds = set(
            individual.loc[
                (individual["split"] == split) & (individual["experiment"] == experiment), "seed"
            ]
        )
        missing = sorted(EXPECTED_SEEDS - seeds)
        if missing and args.require_all_seeds:
            raise SystemExit(f"Missing seeds for {experiment}/{split}: {missing}")
        aggregated.loc[
            (aggregated["split"] == split) & (aggregated["experiment"] == experiment), "missing_seeds"
        ] = ",".join(map(str, missing))
    aggregated.to_csv(output_dir / "seed_summary.csv", index=False)

    model_pairs = [
        ("m1_distilbert_ce", "m2_distilbert_rdrop", ""),
        ("m1_textcnn_ce", "m2_textcnn_rdrop", "textcnn_"),
    ]
    for m1_name, m2_name, filename_prefix in model_pairs:
        m1 = individual.loc[individual["experiment"] == m1_name]
        m2 = individual.loc[individual["experiment"] == m2_name]
        paired = m2.merge(m1, on=["split", "seed"], suffixes=("_rdrop", "_ce"))
        if paired.empty:
            continue
        for metric in METRICS:
            paired[f"{metric}_delta"] = paired[f"{metric}_rdrop"] - paired[f"{metric}_ce"]
        delta_columns = ["split", "seed"] + [f"{metric}_delta" for metric in METRICS]
        paired[delta_columns].to_csv(output_dir / f"{filename_prefix}rdrop_minus_ce_paired.csv", index=False)
        paired_summary = (
            paired.groupby("split")[[f"{metric}_delta" for metric in METRICS]]
            .agg(["count", "mean", "std"])
            .reset_index()
        )
        paired_summary.to_csv(output_dir / f"{filename_prefix}rdrop_minus_ce_summary.csv", index=False)

    print(individual[["split", "experiment", "seed", "macro_f1", "oos_f1"]].to_string(index=False))
    print(f"Summary: {output_dir}")


if __name__ == "__main__":
    main()
