"""Audit the personalized PPTX, speaker script, media, and disclosure consistency."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DELIVERY = PROJECT_ROOT.parent / "delivery"
PPTX = DELIVERY / "MFE310TC_Presentation/MFE310TC_Presentation.pptx"
DOCX = DELIVERY / "MFE310TC_Presentation_Speaker_Script.docx"
NS = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main", "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def seconds(value: str) -> int:
    minutes, seconds = map(int, value.split(":"))
    return minutes * 60 + seconds


def main() -> None:
    errors = []
    checks = 0

    def check(condition: bool, message: str) -> None:
        nonlocal checks
        checks += 1
        if not condition:
            errors.append(message)

    with zipfile.ZipFile(PPTX) as archive:
        slide_names = sorted(
            [name for name in archive.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)],
            key=lambda name: int(re.search(r"(\d+)", Path(name).stem).group(1)),
        )
        check(len(slide_names) == 8, f"Expected 8 slides, found {len(slide_names)}")
        slides = []
        for name in slide_names:
            root = ET.fromstring(archive.read(name))
            slides.append(" ".join(node.text or "" for node in root.findall(".//a:t", NS)))
        embedded_hashes = {
            sha256_bytes(archive.read(name))
            for name in archive.namelist()
            if name.startswith("ppt/media/") and not name.endswith("/")
        }

    with zipfile.ZipFile(DOCX) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
        script_text = " ".join(node.text or "" for node in root.findall(".//w:t", NS))

    all_slide_text = " ".join(slides)
    check("STUDENT_NAME_REDACTED" in slides[0], "Student name missing on slide 1")
    check("STUDENT_ID_REDACTED" in slides[0], "Student ID missing on slide 1")
    check("Individual project (one-person group)" in slides[0], "One-person group statement missing on slide 1")
    check("TO BE ADDED" not in all_slide_text, "Unfilled student/group placeholder remains")
    check("INDIVIDUAL PROJECT" in slides[7], "Individual contribution heading missing on slide 8")
    check("I am responsible for the full submission" in slides[7], "Individual ownership statement missing")
    check("AI assistance is declared in the report" in slides[7], "AI declaration missing on slide 8")
    check("EDIT BEFORE RECORDING" not in all_slide_text, "Outdated contribution-editing notice remains")

    expected_times = ["0:00", "0:20", "0:55", "1:35", "2:15", "2:50", "3:25", "4:05", "4:40"]
    for index in range(8):
        check(expected_times[index] in script_text and expected_times[index + 1] in script_text, f"Timing mismatch on slide {index + 1}")
    total_seconds = seconds(expected_times[-1]) - seconds(expected_times[0])
    check(total_seconds == 280, f"Target duration must be 280 seconds, found {total_seconds}")
    check(total_seconds <= 300, "Presentation exceeds five minutes")
    check("598 words" in script_text, "Updated spoken word count missing from speaker script")
    check("As a one-person group" in script_text, "Speaker script does not reflect individual project")
    check("AI assistance as declared in the report" in script_text, "Speaker script AI declaration missing")

    required_metrics = {
        3: ["0.8508", "0.5862", "0.4176"],
        4: ["+0.0050", "+0.0057", "+0.0021", "+0.0072", "0.0117"],
        5: ["0.7131", "0.7210", "0.5921", "0.6034", "0.7898", "0.7246"],
    }
    for index, values in required_metrics.items():
        for value in values:
            check(value in slides[index], f"Metric {value} missing on slide {index + 1}")
    for value in ["0.939", "0.932", "0.778"]:
        check(value in script_text, f"Inference confidence {value} missing from speaker script")

    check("approved remote Linux GPU practice is not claimed" in slides[2], "GPU disclosure missing on slide 3")
    check("approved remote Linux GPU requirement not met" in slides[7], "GPU limitation missing on slide 8")

    source_hashes = {
        sha256_bytes(path.read_bytes())
        for path in [
            PROJECT_ROOT / "results/data_checks/figures/split_counts.png",
            PROJECT_ROOT / "results/report_assets/final_model_comparison.png",
            PROJECT_ROOT / "results/report_assets/final_noise_comparison.png",
            PROJECT_ROOT / "results/report_assets/rdrop_paired_effects.png",
            PROJECT_ROOT / "results/report_assets/inference_demo.png",
        ]
    }
    check(embedded_hashes == source_hashes, "Embedded PPTX media differ from the five verified source images")

    inference = json.loads((PROJECT_ROOT / "results/report_assets/inference_demo.json").read_text(encoding="utf-8"))
    check(len(inference) == 3, "Inference demo must contain three actual examples")
    check(inference[0]["predicted_label"] == "traffic" and inference[0]["correct"], "Supported-intent demo changed")
    check(inference[1]["predicted_label"] == "oos" and inference[1]["correct"], "OOS demo changed")
    check(inference[2]["predicted_label"] == "play_music" and not inference[2]["correct"], "Error demo changed")

    summary = {
        "status": "passed" if not errors else "failed",
        "checks_performed": checks,
        "slides": len(slides),
        "target_seconds": total_seconds,
        "student": "STUDENT_NAME_REDACTED",
        "student_id": "STUDENT_ID_REDACTED",
        "group": "Individual project (one-person group)",
        "inference_examples": len(inference),
        "errors": errors,
    }
    output = PROJECT_ROOT / "results/audit/presentation_audit_summary.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
