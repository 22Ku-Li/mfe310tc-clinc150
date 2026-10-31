"""Load, validate, and audit the official CLINC150 full dataset.

The original train/validation/test split is preserved. OOS examples are added
as a 151st class; no sampling or model-driven filtering is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OOS_LABEL = "oos"
SPLIT_SOURCES = {
    "train": ("train", "oos_train"),
    "val": ("val", "oos_val"),
    "test": ("test", "oos_test"),
}
EXPECTED_SPLIT_COUNTS = {"train": 15100, "val": 3100, "test": 5500}
SOURCE_URL = "https://archive.ics.uci.edu/static/public/570/clinc150.zip"
DATASET_DOI = "https://doi.org/10.24432/C5MP58"
OFFICIAL_REPOSITORY_COMMIT = "828f8093932c8fe6ca7936c3d2e52903b1c523de"


def sha256(path: Path) -> str:
    """Return a file's SHA-256 digest without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_text(text: str) -> str:
    """Normalize text only for duplicate auditing, not for model input."""
    return re.sub(r"\s+", " ", text.strip().lower())


def load_clinc_data(raw_path: Path) -> pd.DataFrame:
    """Load CLINC150 JSON into a validated, stably indexed DataFrame."""
    with raw_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    missing_keys = [key for keys in SPLIT_SOURCES.values() for key in keys if key not in raw]
    if missing_keys:
        raise ValueError(f"Missing expected CLINC150 keys: {missing_keys}")

    records: list[dict[str, Any]] = []
    for split, source_keys in SPLIT_SOURCES.items():
        for source_key in source_keys:
            for row_index, row in enumerate(raw[source_key]):
                if not isinstance(row, list) or len(row) != 2:
                    raise ValueError(f"Invalid row at {source_key}[{row_index}]: {row!r}")
                text, label = row
                records.append(
                    {
                        "sample_id": f"{split}_{source_key}_{row_index:05d}",
                        "split": split,
                        "source_key": source_key,
                        "source_index": row_index,
                        "text": text,
                        "label": label,
                    }
                )

    frame = pd.DataFrame.from_records(records)
    if frame[["text", "label"]].isna().any().any():
        raise ValueError("Dataset contains missing text or labels")
    if not frame["text"].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("Dataset contains empty or non-string text")
    if not frame["label"].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("Dataset contains empty or non-string labels")
    if frame["sample_id"].duplicated().any():
        raise ValueError("Sample IDs are not unique")

    actual_counts = frame.groupby("split").size().to_dict()
    if actual_counts != EXPECTED_SPLIT_COUNTS:
        raise ValueError(f"Unexpected split counts: {actual_counts}")

    in_scope_labels = set(frame.loc[frame["label"] != OOS_LABEL, "label"])
    if len(in_scope_labels) != 150:
        raise ValueError(f"Expected 150 in-scope labels, found {len(in_scope_labels)}")

    frame["normalized_text"] = frame["text"].map(normalize_text)
    frame["word_count"] = frame["text"].str.split().str.len().astype(int)
    frame["character_count"] = frame["text"].str.len().astype(int)
    return frame


def length_statistics(frame: pd.DataFrame) -> dict[str, float | int]:
    """Summarize utterance length using simple whitespace token counts."""
    values = frame["word_count"].to_numpy()
    return {
        "minimum": int(values.min()),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p95": float(np.percentile(values, 95)),
        "maximum": int(values.max()),
    }


def audit_data(frame: pd.DataFrame, results_dir: Path) -> dict[str, Any]:
    """Write split, quality, distribution, and leakage audit artifacts."""
    results_dir.mkdir(parents=True, exist_ok=True)
    split_rows: list[dict[str, Any]] = []
    for split in SPLIT_SOURCES:
        group = frame.loc[frame["split"] == split]
        split_rows.append(
            {
                "split": split,
                "rows": len(group),
                "in_scope_rows": int((group["label"] != OOS_LABEL).sum()),
                "oos_rows": int((group["label"] == OOS_LABEL).sum()),
                "unique_labels": int(group["label"].nunique()),
                "unique_texts": int(group["text"].nunique()),
                "missing_text": int(group["text"].isna().sum()),
                "missing_labels": int(group["label"].isna().sum()),
                "empty_text": int((group["text"].str.strip() == "").sum()),
                "duplicate_text_extra_rows": int(group["text"].duplicated().sum()),
                "duplicate_text_label_extra_rows": int(group[["text", "label"]].duplicated().sum()),
                "mean_words": float(group["word_count"].mean()),
                "p95_words": float(np.percentile(group["word_count"], 95)),
                "max_words": int(group["word_count"].max()),
            }
        )
    split_summary = pd.DataFrame(split_rows)
    split_summary.to_csv(results_dir / "split_summary.csv", index=False)

    distribution = (
        frame.groupby(["split", "label"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["split", "label"], kind="stable")
    )
    distribution.to_csv(results_dir / "intent_distribution.csv", index=False)

    overlap_rows: list[dict[str, Any]] = []
    split_names = list(SPLIT_SOURCES)
    for index, split_a in enumerate(split_names):
        for split_b in split_names[index + 1 :]:
            group_a = frame.loc[frame["split"] == split_a]
            group_b = frame.loc[frame["split"] == split_b]
            for kind, column in (("exact", "text"), ("normalized", "normalized_text")):
                values_a = set(group_a[column])
                values_b = set(group_b[column])
                overlap = sorted(values_a & values_b)
                overlap_rows.append(
                    {
                        "split_a": split_a,
                        "split_b": split_b,
                        "kind": kind,
                        "shared_unique_texts": len(overlap),
                        "examples": " ||| ".join(overlap[:10]),
                    }
                )
    pd.DataFrame(overlap_rows).to_csv(results_dir / "cross_split_overlaps.csv", index=False)

    conflicting_texts = (
        frame.groupby("normalized_text")["label"].nunique().loc[lambda series: series > 1].index
    )
    conflicts = frame.loc[frame["normalized_text"].isin(conflicting_texts)].sort_values(
        ["normalized_text", "label", "split"], kind="stable"
    )
    conflicts.to_csv(results_dir / "conflicting_label_examples.csv", index=False)

    train_in_scope_counts = distribution.loc[
        (distribution["split"] == "train") & (distribution["label"] != OOS_LABEL), "count"
    ]
    summary: dict[str, Any] = {
        "dataset": "CLINC150 data_full",
        "total_rows": len(frame),
        "in_scope_label_count": int(frame.loc[frame["label"] != OOS_LABEL, "label"].nunique()),
        "includes_oos_as_151st_class": True,
        "split_summary": split_rows,
        "train_in_scope_examples_per_intent": {
            "minimum": int(train_in_scope_counts.min()),
            "maximum": int(train_in_scope_counts.max()),
            "mean": float(train_in_scope_counts.mean()),
        },
        "length_statistics_by_split": {
            split: length_statistics(frame.loc[frame["split"] == split]) for split in SPLIT_SOURCES
        },
        "cross_split_overlaps": overlap_rows,
        "normalized_texts_with_conflicting_labels": int(len(conflicting_texts)),
        "leakage_policy": (
            "Original splits are fixed. Vectorizers and classifiers are fitted on train only. "
            "Validation is for model selection; final test evaluation is run only after configuration freeze."
        ),
    }
    (results_dir / "data_quality_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    figure_dir = results_dir / "figures"
    figure_dir.mkdir(exist_ok=True)
    plot_split_counts(split_summary, figure_dir / "split_counts.png")
    plot_train_lengths(frame.loc[frame["split"] == "train"], figure_dir / "train_text_lengths.png")
    return summary


def plot_split_counts(split_summary: pd.DataFrame, output_path: Path) -> None:
    """Plot fixed split sizes with OOS shown separately."""
    ordered = split_summary.set_index("split").loc[["train", "val", "test"]]
    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.bar(ordered.index, ordered["in_scope_rows"], label="In-scope")
    axis.bar(ordered.index, ordered["oos_rows"], bottom=ordered["in_scope_rows"], label="OOS")
    axis.set_ylabel("Number of utterances")
    axis.set_title("CLINC150 fixed split sizes")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_train_lengths(train: pd.DataFrame, output_path: Path) -> None:
    """Plot training-only utterance lengths for preprocessing decisions."""
    fig, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.hist(train["word_count"], bins=range(1, int(train["word_count"].max()) + 3), color="#3568b8")
    axis.set_xlabel("Whitespace-delimited words per utterance")
    axis.set_ylabel("Training utterances")
    axis.set_title("Training text-length distribution")
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=PROJECT_ROOT / "data/raw/data_full.json")
    parser.add_argument("--processed-dir", type=Path, default=PROJECT_ROOT / "data/processed")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results/data_checks")
    args = parser.parse_args()

    raw_path = args.raw.resolve()
    processed_dir = args.processed_dir.resolve()
    results_dir = args.results_dir.resolve()
    processed_dir.mkdir(parents=True, exist_ok=True)

    frame = load_clinc_data(raw_path)
    for split in SPLIT_SOURCES:
        output = frame.loc[frame["split"] == split].drop(columns=["normalized_text"])
        output.to_csv(processed_dir / f"{split}.csv", index=False)

    summary = audit_data(frame, results_dir)
    zip_path = raw_path.with_name("clinc150.zip")
    manifest = {
        "dataset": "CLINC150",
        "version": "data_full",
        "source_url": SOURCE_URL,
        "dataset_doi": DATASET_DOI,
        "official_repository": "https://github.com/clinc/oos-eval",
        "official_repository_commit": OFFICIAL_REPOSITORY_COMMIT,
        "license": "UCI metadata: CC BY 4.0; bundled LICENSE: Attribution 3.0 Unported (discrepancy documented)",
        "license_file": "data/raw/LICENSE",
        "license_note": "The UCI webpage and bundled license file report different versions; both are preserved with attribution.",
        "raw_file": str(raw_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "raw_sha256": sha256(raw_path),
        "archive_sha256": sha256(zip_path) if zip_path.exists() else None,
        "processed_at_utc": datetime.now(timezone.utc).isoformat(),
        "row_count": len(frame),
        "split_counts": {row["split"]: row["rows"] for row in summary["split_summary"]},
    }
    (results_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "status": "success",
                "rows": len(frame),
                "split_counts": manifest["split_counts"],
                "in_scope_labels": summary["in_scope_label_count"],
                "normalized_conflicts": summary["normalized_texts_with_conflicting_labels"],
                "results_dir": str(results_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
