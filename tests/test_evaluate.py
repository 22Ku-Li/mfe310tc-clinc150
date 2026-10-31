"""Unit tests for the shared classification evaluation contract."""

import unittest

import numpy as np

from src.evaluate import compute_classification_metrics


class EvaluationTests(unittest.TestCase):
    def test_metrics_and_oos_score(self) -> None:
        y_true = ["a", "a", "b", "oos"]
        y_pred = ["a", "b", "b", "oos"]
        metrics = compute_classification_metrics(
            y_true,
            y_pred,
            labels=["a", "b", "oos"],
            oos_label="oos",
            oos_scores=np.array([0.1, 0.2, 0.2, 0.9]),
        )
        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["oos_precision"], 1.0)
        self.assertAlmostEqual(metrics["oos_recall"], 1.0)
        self.assertAlmostEqual(metrics["oos_f1"], 1.0)
        self.assertAlmostEqual(metrics["oos_auroc"], 1.0)
        self.assertEqual(metrics["rows"], 4)
        self.assertEqual(metrics["in_scope_rows"], 3)

    def test_empty_prediction_is_safe(self) -> None:
        metrics = compute_classification_metrics(
            ["a", "oos"], ["b", "b"], labels=["a", "b", "oos"], oos_label="oos"
        )
        self.assertEqual(metrics["accuracy"], 0.0)
        self.assertEqual(metrics["oos_f1"], 0.0)
        self.assertIsNone(metrics["oos_auroc"])


if __name__ == "__main__":
    unittest.main()
