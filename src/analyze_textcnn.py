"""Create final TextCNN comparison figures, summaries, and error transitions."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import PROJECT_ROOT


def main() -> None:
    output = PROJECT_ROOT / "results/report_assets"
    output.mkdir(parents=True, exist_ok=True)
    clean = pd.read_csv(PROJECT_ROOT / "results/summary/seed_summary.csv")
    test = clean.loc[clean["split"] == "test"].set_index("experiment")
    baseline = json.loads((PROJECT_ROOT / "results/m0_tfidf_logreg/test/metrics.json").read_text(encoding="utf-8"))
    noise_summary = pd.read_csv(PROJECT_ROOT / "results/summary/robustness_seed_summary.csv")
    baseline_noise = pd.read_csv(PROJECT_ROOT / "results/robustness/m0_tfidf_logreg/test/noise_metrics.csv")

    metrics = ["accuracy", "macro_f1", "in_scope_macro_f1", "oos_recall", "oos_f1", "oos_auroc"]
    labels = ["Accuracy", "Macro-F1", "In-scope\nMacro-F1", "OOS recall", "OOS F1", "OOS AUROC"]
    model_specs = [("M0 TF-IDF", "#6b7280"), ("M1 TextCNN", "#3568b8"), ("M2 R-Drop", "#4c8c78")]
    x = np.arange(len(metrics))
    width = 0.25
    fig, axis = plt.subplots(figsize=(9.0, 4.8))
    for model_index, (model_name, color) in enumerate(model_specs):
        if model_name.startswith("M0"):
            values = np.array([baseline[metric] for metric in metrics])
            errors = None
        else:
            experiment = "m1_textcnn_ce" if model_name.startswith("M1") else "m2_textcnn_rdrop"
            row = test.loc[experiment]
            values = np.array([row[f"{metric}_mean"] for metric in metrics])
            errors = np.array([row[f"{metric}_std"] for metric in metrics])
        axis.bar(x + (model_index - 1) * width, values, width, yerr=errors, capsize=3, label=model_name, color=color)
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 1.03)
    axis.set_ylabel("Final test score")
    axis.set_title("Frozen-model comparison on CLINC150 test data")
    axis.legend(frameon=False, ncol=3, loc="upper center")
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output / "final_model_comparison.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.8, 4.7))
    axis.plot(baseline_noise["noise_rate"] * 100, baseline_noise["macro_f1"], marker="o", linewidth=2, label="M0 TF-IDF", color="#6b7280")
    for experiment, label, color in [("m1_textcnn_ce", "M1 TextCNN", "#3568b8"), ("m2_textcnn_rdrop", "M2 R-Drop", "#4c8c78")]:
        subset = noise_summary.loc[(noise_summary["split"] == "test") & (noise_summary["experiment"] == experiment)].sort_values("noise_rate")
        axis.errorbar(subset["noise_rate"] * 100, subset["macro_f1_mean"], yerr=subset["macro_f1_std"], marker="o", linewidth=2, capsize=4, label=label, color=color)
    axis.set_xlabel("Deterministic spelling-noise rate (%)")
    axis.set_ylabel("Final test Macro-F1")
    axis.set_title("Robustness comparison with mean ± seed SD")
    axis.set_ylim(0, 1.0)
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output / "final_noise_comparison.png", dpi=180)
    plt.close(fig)

    paired_clean = pd.read_csv(PROJECT_ROOT / "results/summary/textcnn_rdrop_minus_ce_paired.csv")
    paired_clean = paired_clean.loc[paired_clean["split"] == "test"]
    paired_noise = pd.read_csv(PROJECT_ROOT / "results/summary/textcnn_robustness_paired.csv")
    paired_noise = paired_noise.loc[paired_noise["split"] == "test"]
    fig, axis = plt.subplots(figsize=(7.8, 4.6))
    positions = np.arange(3)
    axis.bar(positions - 0.2, paired_clean.sort_values("seed")["macro_f1_delta"], width=0.2, label="Clean", color="#3568b8")
    for index, rate in enumerate([0.05, 0.10]):
        values = paired_noise.loc[paired_noise["noise_rate"] == rate].sort_values("seed")["macro_f1_delta"]
        axis.bar(positions + index * 0.2, values, width=0.2, label=f"Noise {rate:.0%}")
    axis.axhline(0, color="black", linewidth=0.7)
    axis.set_xticks(positions, ["Seed 13", "Seed 42", "Seed 87"])
    axis.set_ylabel("M2 − M1 Macro-F1")
    axis.set_title("Paired R-Drop effect on final test data")
    axis.legend(frameon=False, ncol=3)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output / "rdrop_paired_effects.png", dpi=180)
    plt.close(fig)

    m1_predictions = pd.read_csv(PROJECT_ROOT / "results/m1_textcnn_ce/seed_42/test/predictions.csv")
    m2_predictions = pd.read_csv(PROJECT_ROOT / "results/m2_textcnn_rdrop/seed_42/test/predictions.csv")
    columns = ["sample_id", "text", "true_label", "predicted_label", "predicted_probability", "oos_score", "correct"]
    merged = m1_predictions[columns].merge(m2_predictions[columns], on="sample_id", suffixes=("_m1", "_m2"))
    merged["true_label"] = merged["true_label_m1"]
    corrected = merged.loc[(~merged["correct_m1"]) & (merged["correct_m2"])].nlargest(20, "predicted_probability_m2")
    introduced = merged.loc[(merged["correct_m1"]) & (~merged["correct_m2"])].nlargest(20, "predicted_probability_m2")
    shared = merged.loc[(~merged["correct_m1"]) & (~merged["correct_m2"])].head(30)
    corrected.to_csv(output / "errors_corrected_by_rdrop_seed42.csv", index=False)
    introduced.to_csv(output / "errors_introduced_by_rdrop_seed42.csv", index=False)
    shared.to_csv(output / "shared_errors_seed42.csv", index=False)

    summary = {
        "m0_test": baseline,
        "m1_test_mean": {metric: float(test.loc["m1_textcnn_ce", f"{metric}_mean"]) for metric in metrics},
        "m1_test_std": {metric: float(test.loc["m1_textcnn_ce", f"{metric}_std"]) for metric in metrics},
        "m2_test_mean": {metric: float(test.loc["m2_textcnn_rdrop", f"{metric}_mean"]) for metric in metrics},
        "m2_test_std": {metric: float(test.loc["m2_textcnn_rdrop", f"{metric}_std"]) for metric in metrics},
        "clean_macro_f1_delta_by_seed": {str(int(row.seed)): float(row.macro_f1_delta) for row in paired_clean.itertuples()},
        "noise_macro_f1_mean": {
            experiment: {
                str(float(row.noise_rate)): float(row.macro_f1_mean)
                for row in noise_summary.loc[(noise_summary["split"] == "test") & (noise_summary["experiment"] == experiment)].itertuples()
            }
            for experiment in ["m1_textcnn_ce", "m2_textcnn_rdrop"]
        },
        "baseline_noise_macro_f1": {str(float(row.noise_rate)): float(row.macro_f1) for row in baseline_noise.itertuples()},
        "approved_remote_linux_gpu_used": False,
    }
    (output / "final_analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
