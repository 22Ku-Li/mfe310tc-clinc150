"""Shared evaluation utilities for intent classification and OOS detection."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: Iterable[str],
    y_pred: Iterable[str],
    labels: list[str],
    oos_label: str = "oos",
    oos_scores: np.ndarray | None = None,
) -> dict[str, float | int | None]:
    """Compute overall, in-scope, and OOS metrics on one fixed split."""
    true = np.asarray(list(y_true))
    pred = np.asarray(list(y_pred))
    in_scope_labels = [label for label in labels if label != oos_label]
    in_scope_mask = true != oos_label

    oos_precision, oos_recall, oos_f1, _ = precision_recall_fscore_support(
        true == oos_label,
        pred == oos_label,
        average="binary",
        pos_label=True,
        zero_division=0,
    )
    metrics: dict[str, float | int | None] = {
        "rows": int(len(true)),
        "accuracy": float(accuracy_score(true, pred)),
        "macro_precision": float(precision_score(true, pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(true, pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(true, pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(true, pred, labels=labels, average="weighted", zero_division=0)),
        "in_scope_rows": int(in_scope_mask.sum()),
        "in_scope_accuracy": float(accuracy_score(true[in_scope_mask], pred[in_scope_mask])),
        "in_scope_macro_f1": float(
            f1_score(
                true[in_scope_mask],
                pred[in_scope_mask],
                labels=in_scope_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "oos_rows": int((true == oos_label).sum()),
        "oos_precision": float(oos_precision),
        "oos_recall": float(oos_recall),
        "oos_f1": float(oos_f1),
        "oos_auroc": None,
    }
    if oos_scores is not None:
        metrics["oos_auroc"] = float(roc_auc_score(true == oos_label, np.asarray(oos_scores)))
    return metrics


def save_classification_report(
    y_true: Iterable[str], y_pred: Iterable[str], labels: list[str], output_path: Path
) -> None:
    """Save per-intent precision, recall, F1, and support."""
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(output_path, index_label="label")


def save_top_confusions(
    y_true: Iterable[str], y_pred: Iterable[str], labels: list[str], output_path: Path, limit: int = 50
) -> None:
    """Save the most frequent off-diagonal confusion pairs."""
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    rows = []
    for true_index, true_label in enumerate(labels):
        for pred_index, pred_label in enumerate(labels):
            if true_index != pred_index and matrix[true_index, pred_index] > 0:
                rows.append(
                    {
                        "true_label": true_label,
                        "predicted_label": pred_label,
                        "count": int(matrix[true_index, pred_index]),
                    }
                )
    pairs = pd.DataFrame(rows, columns=["true_label", "predicted_label", "count"])
    pairs = pairs.sort_values(["count", "true_label", "predicted_label"], ascending=[False, True, True])
    pairs.head(limit).to_csv(output_path, index=False)
