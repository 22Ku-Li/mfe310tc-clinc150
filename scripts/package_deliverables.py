"""Package the current report, public data, code, and reproducibility evidence."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DELIVERY_DIR = PROJECT_ROOT.parent / "delivery"
CODE_ZIP = DELIVERY_DIR / "MFE310TC_Code_Data_and_Results.zip"
REPORT_PDF = DELIVERY_DIR / "MFE310TC_Technical_Report_CPU_REVIEW.pdf"
PRESENTATION_ZIP = DELIVERY_DIR / "MFE310TC_Presentation_Package.zip"
COMBINED_ZIP = DELIVERY_DIR / "MFE310TC_Current_Deliverables.zip"
PACKAGE_ROOT = "MFE310TC_CLINC150_Project"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_files() -> list[Path]:
    root_files = [
        ".gitignore",
        "README.md",
        "requirements.txt",
        "requirements-lock.txt",
        "requirements-transformers.txt",
        "requirements-cpu.txt",
        "requirements-demo.txt",
    ]
    files = [PROJECT_ROOT / name for name in root_files]
    for directory in ["configs", "scripts", "src", "tests", "report", "audit", "results", "data/processed"]:
        files.extend(
            path
            for path in (PROJECT_ROOT / directory).rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix not in {".pyc", ".pyo"}
            and not path.name.startswith(".")
        )
    demo_artifact = PROJECT_ROOT / "artifacts/m0_tfidf_logreg_retrained.joblib"
    if demo_artifact.is_file():
        files.append(demo_artifact)
    files.extend(
        PROJECT_ROOT / "data/raw" / name
        for name in ["data_full.json", "LICENSE", "meta.txt"]
    )
    return sorted(set(files))


def main() -> None:
    DELIVERY_DIR.mkdir(exist_ok=True)
    files = selected_files()
    for required in [REPORT_PDF, PRESENTATION_ZIP, DELIVERY_DIR / "START_HERE.txt", *files]:
        if not required.exists():
            raise FileNotFoundError(required)

    manifest = {
        "package_status": "technical_results_complete_admin_review_required",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "public_dataset_license": "UCI metadata: CC BY 4.0; bundled LICENSE: Attribution 3.0 Unported; discrepancy documented and original file included",
        "official_test_metrics_present": True,
        "m1_m2_formal_cpu_results_present": True,
        "approved_remote_linux_gpu_used": False,
        "files_sha256": {
            str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): sha256(path)
            for path in files
        },
    }
    text_suffixes = {".py", ".md", ".txt", ".yaml", ".yml", ".json", ".csv", ".sh", ".gitignore"}
    for path in files:
        if (path.suffix in text_suffixes or path.name == ".gitignore") and path.name != "package_deliverables.py":
            content = path.read_text(encoding="utf-8", errors="ignore")
            # Decode JSON-style escaped backslashes before scanning for drive paths.
            normalized = content.replace("\\\\", "\\")
            if re.search(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/][^\s\"']+", normalized):
                raise ValueError(f"Potential Windows absolute path found in {path}")

    with zipfile.ZipFile(CODE_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            relative = str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
            archive.write(path, f"{PACKAGE_ROOT}/{relative}")
        archive.writestr(
            f"{PACKAGE_ROOT}/PACKAGE_MANIFEST.json",
            json.dumps(manifest, indent=2, ensure_ascii=False),
        )

    checksums = (
        f"{sha256(REPORT_PDF)}  {REPORT_PDF.name}\n"
        f"{sha256(CODE_ZIP)}  {CODE_ZIP.name}\n"
        f"{sha256(PRESENTATION_ZIP)}  {PRESENTATION_ZIP.name}\n"
    )
    (DELIVERY_DIR / "SHA256SUMS.txt").write_bytes(checksums.encode("utf-8"))
    with zipfile.ZipFile(COMBINED_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in [
            DELIVERY_DIR / "START_HERE.txt",
            REPORT_PDF,
            CODE_ZIP,
            PRESENTATION_ZIP,
            DELIVERY_DIR / "report_build_info.json",
            DELIVERY_DIR / "SHA256SUMS.txt",
        ]:
            archive.write(path, path.name)

    for path in [CODE_ZIP, COMBINED_ZIP]:
        with zipfile.ZipFile(path) as archive:
            corrupt = archive.testzip()
            if corrupt:
                raise ValueError(f"Corrupt ZIP member in {path}: {corrupt}")
            member_count = len(archive.namelist())
        print(f"{path.name}: {path.stat().st_size:,} bytes, {member_count} members, sha256={sha256(path)}")


if __name__ == "__main__":
    main()
