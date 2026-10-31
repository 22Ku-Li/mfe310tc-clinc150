"""Download and verify the pinned UCI CLINC150 full dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = "https://archive.ics.uci.edu/static/public/570/clinc150.zip"
EXPECTED_ARCHIVE_SHA256 = "0d8ecc3e1edd7b25cabde0177544ce536ddf773844bc80ef1a75f36e7f030ea2"
EXPECTED_RAW_SHA256 = "36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0"
ARCHIVE_MEMBERS = {
    "clinc150_uci/data_full.json": "data_full.json",
    "clinc150_uci/LICENSE": "LICENSE",
    "clinc150_uci/meta.txt": "meta.txt",
}


def file_sha256(path: Path) -> str:
    """Compute a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path, expected_sha256: str) -> None:
    """Fail loudly if a downloaded or cached file differs from the pinned version."""
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise ValueError(f"SHA-256 mismatch for {path}: expected {expected_sha256}, found {actual}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Redownload the archive even if it is valid")
    args = parser.parse_args()

    raw_dir = PROJECT_ROOT / "data/raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive_path = raw_dir / "clinc150.zip"

    if args.force or not archive_path.exists() or file_sha256(archive_path) != EXPECTED_ARCHIVE_SHA256:
        temporary_path = archive_path.with_suffix(".zip.part")
        with urllib.request.urlopen(SOURCE_URL, timeout=60) as response, temporary_path.open("wb") as output:
            shutil.copyfileobj(response, output)
        verify(temporary_path, EXPECTED_ARCHIVE_SHA256)
        temporary_path.replace(archive_path)
    else:
        verify(archive_path, EXPECTED_ARCHIVE_SHA256)

    with zipfile.ZipFile(archive_path) as archive:
        for member, filename in ARCHIVE_MEMBERS.items():
            (raw_dir / filename).write_bytes(archive.read(member))

    raw_path = raw_dir / "data_full.json"
    verify(raw_path, EXPECTED_RAW_SHA256)
    print(
        json.dumps(
            {
                "status": "success",
                "source": SOURCE_URL,
                "archive_sha256": EXPECTED_ARCHIVE_SHA256,
                "raw_sha256": EXPECTED_RAW_SHA256,
                "raw_file": str(raw_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
