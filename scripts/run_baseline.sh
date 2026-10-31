#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ "${ALLOW_OVERWRITE_RESULTS:-0}" != "1" && -d results/m0_tfidf_logreg ]]; then
  echo "Existing baseline results found. Work in a separate copy and set ALLOW_OVERWRITE_RESULTS=1 only if replacement is intended." >&2
  exit 1
fi
if [[ ! -f data/raw/data_full.json ]]; then
  python -m src.download_data
fi
python -m src.data
python -m src.train_baseline --config configs/baseline.yaml --split val
