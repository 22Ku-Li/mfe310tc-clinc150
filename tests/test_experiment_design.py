"""Tests for the controlled M1/M2 experiment-design gate."""

import tempfile
import unittest
from pathlib import Path

import yaml

from src.config import load_config
from src.data import PROJECT_ROOT
from src.validate_experiment_design import compare_configs


class ExperimentDesignTests(unittest.TestCase):
    def test_repository_configs_are_controlled(self) -> None:
        report = compare_configs(
            PROJECT_ROOT / "configs/distilbert.yaml", PROJECT_ROOT / "configs/rdrop.yaml"
        )
        self.assertEqual(report["status"], "controlled_design_valid")
        self.assertEqual(report["planned_seeds"], [13, 42, 87])

    def test_unexpected_hyperparameter_difference_is_rejected(self) -> None:
        m1_path = PROJECT_ROOT / "configs/distilbert.yaml"
        m2_path = PROJECT_ROOT / "configs/rdrop.yaml"
        modified = load_config(m2_path)
        modified["training"]["learning_rate"] = 0.00003
        with tempfile.TemporaryDirectory() as directory:
            modified_path = Path(directory) / "rdrop_modified.yaml"
            modified_path.write_text(yaml.safe_dump(modified), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unexpected_differences"):
                compare_configs(m1_path, modified_path)


if __name__ == "__main__":
    unittest.main()
