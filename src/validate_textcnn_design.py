"""Write the controlled-design certificate for the TextCNN M1/M2 pair."""

import json

from src.data import PROJECT_ROOT
from src.validate_experiment_design import compare_configs


def main() -> None:
    report = compare_configs(
        PROJECT_ROOT / "configs/textcnn.yaml", PROJECT_ROOT / "configs/textcnn_rdrop.yaml"
    )
    report["execution_note"] = (
        "Formal runs are CPU-reproducible. The approved remote Linux GPU practice requirement is not claimed."
    )
    output = PROJECT_ROOT / "results/experiment_design/textcnn_controlled_comparison.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
