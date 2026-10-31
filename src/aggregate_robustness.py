"""Aggregate multi-seed TextCNN spelling-noise results and paired differences."""

from __future__ import annotations

import json

import pandas as pd

from src.data import PROJECT_ROOT
from src.aggregate_results import METRICS


def main() -> None:
    rows = []
    for path in sorted((PROJECT_ROOT / "results/robustness").glob("m*_textcnn_*/seed_*/noise_*/*/metrics.json")):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        row = {
            "experiment": metrics["experiment"],
            "seed": int(metrics["seed"]),
            "split": metrics["split"],
            "noise_rate": float(metrics["noise_rate"]),
        }
        row.update({metric: metrics.get(metric) for metric in METRICS})
        rows.append(row)
    if not rows:
        raise SystemExit("No TextCNN robustness metrics found")
    runs = pd.DataFrame(rows).sort_values(["split", "noise_rate", "experiment", "seed"])
    output = PROJECT_ROOT / "results/summary"
    output.mkdir(parents=True, exist_ok=True)
    runs.to_csv(output / "robustness_runs.csv", index=False)
    summary = runs.groupby(["split", "experiment", "noise_rate"])[METRICS].agg(["count", "mean", "std"])
    summary.columns = [f"{metric}_{statistic}" for metric, statistic in summary.columns]
    summary.reset_index().to_csv(output / "robustness_seed_summary.csv", index=False)

    m1 = runs.loc[runs["experiment"] == "m1_textcnn_ce"]
    m2 = runs.loc[runs["experiment"] == "m2_textcnn_rdrop"]
    paired = m2.merge(m1, on=["split", "seed", "noise_rate"], suffixes=("_rdrop", "_ce"))
    for metric in METRICS:
        paired[f"{metric}_delta"] = paired[f"{metric}_rdrop"] - paired[f"{metric}_ce"]
    paired.to_csv(output / "textcnn_robustness_paired.csv", index=False)
    paired_summary = paired.groupby(["split", "noise_rate"])[[f"{metric}_delta" for metric in METRICS]].agg(["count", "mean", "std"])
    paired_summary.columns = [f"{metric}_{statistic}" for metric, statistic in paired_summary.columns]
    paired_summary.reset_index().to_csv(output / "textcnn_robustness_summary.csv", index=False)
    print(runs[["split", "noise_rate", "experiment", "seed", "macro_f1", "oos_f1"]].to_string(index=False))
    print(f"Summary: {output}")


if __name__ == "__main__":
    main()
