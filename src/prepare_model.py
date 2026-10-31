"""Materialize one shared pretrained DistilBERT snapshot for all M1/M2 seeds."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from src.config import load_config, resolve_project_path
from src.data import PROJECT_ROOT, sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="configs/distilbert.yaml or configs/rdrop.yaml")
    parser.add_argument("--force", action="store_true", help="Redownload and replace the local base snapshot")
    args = parser.parse_args()

    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:
        raise SystemExit("Install Transformer dependencies with pip install -r requirements-transformers.txt") from error

    config_path = resolve_project_path(args.config)
    config = load_config(config_path)
    model_config = config["model"]
    output_dir = resolve_project_path(model_config["local_model_path"])
    manifest_path = PROJECT_ROOT / "results/model_provenance/base_model_manifest.json"
    if output_dir.exists() and manifest_path.exists() and not args.force:
        print(f"Using existing base model snapshot: {output_dir}")
        print(manifest_path.read_text(encoding="utf-8"))
        return

    source_name = model_config["source_model_name"]
    source_revision = model_config["source_revision"]
    output_dir.mkdir(parents=True, exist_ok=True)
    model = AutoModel.from_pretrained(source_name, revision=source_revision)
    tokenizer = AutoTokenizer.from_pretrained(source_name, revision=source_revision, use_fast=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    file_hashes = {
        str(path.relative_to(output_dir)).replace("\\", "/"): sha256(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    }
    manifest = {
        "source_model_name": source_name,
        "requested_revision": source_revision,
        "resolved_commit": getattr(model.config, "_commit_hash", None),
        "license": "Apache-2.0",
        "source_url": f"https://huggingface.co/{source_name}",
        "local_snapshot": str(output_dir.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "files_sha256": file_hashes,
        "note": (
            "All M1/M2 runs load this same local base snapshot with local_files_only=True. "
            "For long-term reproduction, replace source_revision: main with resolved_commit in the YAML configs."
        ),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
