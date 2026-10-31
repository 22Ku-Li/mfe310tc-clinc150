"""Audit report text, status claims, key figures, and delivery consistency."""

from __future__ import annotations

import json
import re
from pathlib import Path

from pypdf import PdfReader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DELIVERY = PROJECT_ROOT.parent / "delivery"


def main() -> None:
    pdf_path = DELIVERY / "MFE310TC_Technical_Report_CPU_REVIEW.pdf"
    reader = PdfReader(str(pdf_path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    (PROJECT_ROOT / "results/audit/report_text.txt").write_text(text, encoding="utf-8")
    build_info = json.loads((DELIVERY / "report_build_info.json").read_text(encoding="utf-8"))
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    start_here = (DELIVERY / "START_HERE.txt").read_text(encoding="utf-8")
    data_source = (PROJECT_ROOT / "src/data.py").read_text(encoding="utf-8")
    packaging_source = (PROJECT_ROOT / "scripts/package_deliverables.py").read_text(encoding="utf-8")

    errors = []
    checks = 0

    def check(condition: bool, message: str) -> None:
        nonlocal checks
        checks += 1
        if not condition:
            errors.append(message)

    check(len(reader.pages) == 5, f"Expected 5 pages, found {len(reader.pages)}")
    normalized = re.sub(r"\s+", " ", text)
    for value in [
        "0.8508", "0.8293", "0.8343", "0.4373", "0.4227", "0.5862", "0.5746",
        "0.7210", "0.6034", "0.7898", "0.7246", "453,251", "2,875",
    ]:
        check(value in normalized, f"Missing or changed key report value: {value}")
    for required in [
        "approved_remote_linux_gpu_used=false",
        "Attribution 3.0 Unported",
        "CC BY 4.0",
        "AI-use declaration",
        "Kimi Work",
        "thirteen unit/integration tests",
        "R-Drop",
        "TextCNN",
        "STUDENT_NAME_REDACTED",
        "STUDENT_ID_REDACTED",
        "STUDENT_EMAIL_REDACTED",
        "Individual project (one student)",
    ]:
        check(required in normalized, f"Missing required report statement: {required}")
    for forbidden in [
        "M1/M2 GPU RESULTS PENDING",
        "working draft and not a completed submission",
        "No official test-set predictions or test metrics have been generated",
        "The UCI CLINC150 data_full release is used under CC BY 4.0",
    ]:
        check(forbidden not in normalized, f"Outdated or unqualified statement remains: {forbidden}")

    # Independent approximate word count from the rendered PDF body.
    body = text.split("Abstract—", 1)[-1].split("REFERENCES", 1)[0]
    independent_words = re.findall(r"\b[\w’'-]+\b", body)
    check(2250 <= len(independent_words) <= 2750, f"Rendered report body outside word range: {len(independent_words)}")
    check(2250 <= build_info["approximate_body_words_excluding_references_and_appendices"] <= 2750, "Builder body word count is outside the required range")
    check(build_info["official_test_results_present"] is True, "Build status incorrectly says no final results")
    check(build_info["approved_remote_linux_gpu_used"] is False, "Build status incorrectly claims approved GPU")

    # Parameter count is independently recomputed from the architecture.
    parameter_count = 2875 * 100 + sum(width * 100 * 100 + 100 for width in [3, 4, 5]) + 300 * 151 + 151
    check(parameter_count == 453251, f"TextCNN parameter count mismatch: {parameter_count}")

    # Source-of-truth license statements must preserve both versions.
    for source_name, content in [
        ("README", readme),
        ("src/data.py", data_source),
        ("package_deliverables.py", packaging_source),
        ("START_HERE", start_here),
    ]:
        check("Attribution 3.0" in content or "CC BY 3.0" in content, f"Bundled CC BY 3.0 license not acknowledged in {source_name}")
        check("CC BY 4.0" in content or "CC BY4.0" in content, f"UCI CC BY 4.0 metadata not acknowledged in {source_name}")

    # Current README and delivery note must describe completed CPU experiments.
    check("No official test-set predictions or test metrics have been generated" not in readme, "README retains old no-test statement")
    check("thirteen unit/integration tests" in readme, "README test count is outdated")
    check("13 项单元／集成测试通过" in start_here, "Delivery note test count is outdated")
    check("M1 TextCNN" in start_here and "M2 R-Drop" in start_here, "Delivery note model names are inconsistent")

    # Reference entries and in-text citations must cover all five references.
    for number in range(1, 6):
        check(f"[{number}]" in normalized, f"Missing reference/citation marker [{number}]")
    check("Convolutional Neural Networks for Sentence Classification" in normalized, "TextCNN reference missing")
    check("Scikit-learn: Machine Learning in Python" in normalized, "Scikit-learn reference missing")

    summary = {
        "status": "passed" if not errors else "failed",
        "checks_performed": checks,
        "pdf_pages": len(reader.pages),
        "independent_rendered_body_words": len(independent_words),
        "parameter_count_recomputed": parameter_count,
        "errors": errors,
    }
    (PROJECT_ROOT / "results/audit/report_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
