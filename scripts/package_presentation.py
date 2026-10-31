"""Package the editable deck, speaker script, and inference demonstration."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DELIVERY = PROJECT_ROOT.parent / "delivery"
OUTPUT = DELIVERY / "MFE310TC_Presentation_Package.zip"


def main() -> None:
    files = [
        DELIVERY / "MFE310TC_Presentation_Speaker_Script.docx",
        DELIVERY / "START_PRESENTATION.txt",
        PROJECT_ROOT / "scripts/demo_inference.py",
        PROJECT_ROOT / "results/report_assets/inference_demo.json",
        PROJECT_ROOT / "results/audit/presentation_audit_summary.json",
    ]
    deck_root = DELIVERY / "MFE310TC_Presentation"
    files.extend(path for path in deck_root.rglob("*") if path.is_file())
    for path in files:
        if not path.exists():
            raise FileNotFoundError(path)
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            if path.is_relative_to(deck_root):
                target = path.relative_to(deck_root)
            elif path.is_relative_to(PROJECT_ROOT):
                target = path.relative_to(PROJECT_ROOT)
            else:
                target = Path(path.name)
            archive.write(path, str(target).replace("\\", "/"))
    with zipfile.ZipFile(OUTPUT) as archive:
        corrupt = archive.testzip()
        if corrupt:
            raise ValueError(f"Corrupt member: {corrupt}")
    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    print(f"{OUTPUT.name}: {OUTPUT.stat().st_size:,} bytes, sha256={digest}")


if __name__ == "__main__":
    main()
