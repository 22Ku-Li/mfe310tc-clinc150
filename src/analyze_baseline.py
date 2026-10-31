"""Create report-ready baseline figures and representative error tables."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import PROJECT_ROOT


def main() -> None:
    result_dir = PROJECT_ROOT / "results/m0_tfidf_logreg/val"
    figure_dir = result_dir / "figures"
    figure_dir.mkdir(exist_ok=True)
    metrics = json.loads((result_dir / "metrics.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(result_dir / "predictions.csv")
    confusions = pd.read_csv(result_dir / "top_confusions.csv")

    metric_names = ["accuracy", "macro_f1", "in_scope_accuracy", "in_scope_macro_f1", "oos_f1", "oos_auroc"]
    display_names = ["Accuracy", "Macro-F1", "In-scope\naccuracy", "In-scope\nMacro-F1", "OOS F1", "OOS AUROC"]
    values = [metrics[name] for name in metric_names]
    fig, axis = plt.subplots(figsize=(8.4, 4.5))
    bars = axis.bar(display_names, values, color=["#3568b8", "#3568b8", "#4c8c78", "#4c8c78", "#c67a2b", "#c67a2b"])
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Validation score")
    axis.set_title("M0 TF-IDF + logistic regression")
    axis.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(figure_dir / "baseline_metrics.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.5, 4.6))
    bins = np.linspace(0, 1, 31)
    axis.hist(predictions.loc[predictions["true_label"] != "oos", "oos_score"], bins=bins, alpha=0.65, label="In-scope", density=True)
    axis.hist(predictions.loc[predictions["true_label"] == "oos", "oos_score"], bins=bins, alpha=0.65, label="OOS", density=True)
    axis.set_xlabel("Model-estimated OOS probability")
    axis.set_ylabel("Density")
    axis.set_title("OOS score distributions on validation data")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figure_dir / "oos_score_distribution.png", dpi=180)
    plt.close(fig)

    top = confusions.head(10).iloc[::-1]
    labels = top["true_label"] + " → " + top["predicted_label"]
    fig, axis = plt.subplots(figsize=(8.5, 5.2))
    axis.barh(labels, top["count"], color="#8b4f74")
    axis.set_xlabel("Validation errors")
    axis.set_title("Most frequent M0 confusion pairs")
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figure_dir / "top_confusions.png", dpi=180)
    plt.close(fig)

    missed_oos = predictions.loc[(predictions["true_label"] == "oos") & (predictions["predicted_label"] != "oos")].nlargest(15, "predicted_probability")
    false_oos = predictions.loc[(predictions["true_label"] != "oos") & (predictions["predicted_label"] == "oos")].nlargest(15, "oos_score")
    in_scope_errors = predictions.loc[
        (predictions["true_label"] != "oos")
        & (predictions["predicted_label"] != "oos")
        & (~predictions["correct"])
    ].nlargest(25, "predicted_probability")
    missed_oos.assign(error_type="missed_oos").to_csv(result_dir / "errors_missed_oos.csv", index=False)
    false_oos.assign(error_type="false_oos_rejection").to_csv(result_dir / "errors_false_oos.csv", index=False)
    in_scope_errors.assign(error_type="in_scope_confusion").to_csv(result_dir / "errors_in_scope.csv", index=False)

    noise_path = PROJECT_ROOT / "results/robustness/m0_tfidf_logreg/val/noise_metrics.csv"
    if noise_path.exists():
        noise = pd.read_csv(noise_path)
        noise_figure_dir = noise_path.parent / "figures"
        noise_figure_dir.mkdir(exist_ok=True)
        fig, axis = plt.subplots(figsize=(7.4, 4.6))
        for metric, label in [("accuracy", "Accuracy"), ("macro_f1", "Macro-F1"), ("oos_f1", "OOS F1"), ("oos_auroc", "OOS AUROC")]:
            axis.plot(noise["noise_rate"] * 100, noise[metric], marker="o", linewidth=2, label=label)
        axis.set_xlabel("Deterministic spelling-noise rate (%)")
        axis.set_ylabel("Validation score")
        axis.set_ylim(0, 1.02)
        axis.set_title("M0 robustness to spelling noise")
        axis.legend(frameon=False, ncol=2)
        axis.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(noise_figure_dir / "noise_robustness.png", dpi=180)
        plt.close(fig)
    print(f"Figures and error tables: {result_dir}")


if __name__ == "__main__":
    main()
