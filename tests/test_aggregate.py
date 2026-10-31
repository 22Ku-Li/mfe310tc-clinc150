"""Integration test for multi-seed and paired-difference result aggregation."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from src.aggregate_results import METRICS, main as aggregate_main
from src.data import PROJECT_ROOT


class AggregateResultsTests(unittest.TestCase):
    def test_three_seed_summary_and_paired_delta(self) -> None:
        artifacts = PROJECT_ROOT / "artifacts"
        artifacts.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=artifacts) as directory:
            root = Path(directory)
            for experiment, offset in [("m1_distilbert_ce", 0.0), ("m2_distilbert_rdrop", 0.01)]:
                for seed in [13, 42, 87]:
                    output = root / "results" / experiment / f"seed_{seed}" / "val"
                    output.mkdir(parents=True)
                    metrics = {"experiment": experiment, "seed": seed}
                    metrics.update({metric: 0.8 + offset for metric in METRICS})
                    (output / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

            with mock.patch("src.aggregate_results.PROJECT_ROOT", root), mock.patch.object(
                sys, "argv", ["aggregate_results.py", "--require-all-seeds"]
            ):
                aggregate_main()

            summary = pd.read_csv(root / "results/summary/seed_summary.csv")
            paired = pd.read_csv(root / "results/summary/rdrop_minus_ce_paired.csv")
            self.assertEqual(len(summary), 2)
            self.assertEqual(len(paired), 3)
            self.assertAlmostEqual(paired["macro_f1_delta"].mean(), 0.01)


if __name__ == "__main__":
    unittest.main()
