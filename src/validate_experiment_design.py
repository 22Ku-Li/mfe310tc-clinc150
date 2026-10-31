"""Verify that M1 and M2 differ only by the intended R-Drop modification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.config import load_config
from src.data import PROJECT_ROOT

ALLOWED_DIFFERENCES = {
    "experiment_name": "Distinct output names",
    "model.loss": "M1 uses cross_entropy; M2 uses rdrop",
    "rdrop.alpha": "M1 uses 0; M2 uses the single consistency-loss weight",
}


def flatten(config: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested YAML keys for an exact controlled-design comparison."""
    flattened = {}
    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(flatten(value, full_key))
        else:
            flattened[full_key] = value
    return flattened


def compare_configs(m1_path: Path, m2_path: Path) -> dict[str, Any]:
    """Return an auditable design report and reject uncontrolled differences."""
    m1 = flatten(load_config(m1_path))
    m2 = flatten(load_config(m2_path))
    all_keys = sorted(set(m1) | set(m2))
    differences = {
        key: {"m1": m1.get(key), "m2": m2.get(key)}
        for key in all_keys
        if m1.get(key) != m2.get(key)
    }
    unexpected = {key: value for key, value in differences.items() if key not in ALLOWED_DIFFERENCES}
    missing_allowed = sorted(set(ALLOWED_DIFFERENCES) - set(differences))
    if unexpected or missing_allowed:
        raise ValueError(
            json.dumps(
                {"unexpected_differences": unexpected, "expected_but_not_different": missing_allowed},
                indent=2,
            )
        )
    return {
        "status": "controlled_design_valid",
        "m1_config": str(m1_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "m2_config": str(m2_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "allowed_differences": differences,
        "shared_configuration": {
            key: value for key, value in m1.items() if key not in ALLOWED_DIFFERENCES
        },
        "planned_seeds": [13, 42, 87],
        "selection_metric": "validation_macro_f1",
        "test_policy": "Test evaluation is guarded and occurs only after all configurations are frozen",
    }


def main() -> None:
    report = compare_configs(
        PROJECT_ROOT / "configs/distilbert.yaml", PROJECT_ROOT / "configs/rdrop.yaml"
    )
    output_dir = PROJECT_ROOT / "results/experiment_design"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "controlled_comparison.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
