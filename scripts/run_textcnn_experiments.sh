#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Do not accidentally overwrite the supplied historical experiments.
if [[ "${ALLOW_OVERWRITE_RESULTS:-0}" != "1" ]] && { [[ -d results/m1_textcnn_ce ]] || [[ -d results/m2_textcnn_rdrop ]]; }; then
  echo "Existing TextCNN results found. Work in a separate copy and set ALLOW_OVERWRITE_RESULTS=1 only if replacement is intended." >&2
  exit 1
fi

"$PYTHON_BIN" -m src.data
"$PYTHON_BIN" -m src.validate_textcnn_design
for config in configs/textcnn.yaml configs/textcnn_rdrop.yaml; do
  if [[ "$config" == "configs/textcnn.yaml" ]]; then
    experiment="m1_textcnn_ce"
  else
    experiment="m2_textcnn_rdrop"
  fi
  for seed in 13 42 87; do
    mkdir -p "results/$experiment/seed_$seed"
    "$PYTHON_BIN" -m src.train_textcnn \
      --config "$config" \
      --seed "$seed" \
      --device cpu \
      2>&1 | tee "results/$experiment/seed_$seed/cpu_console.log"
  done
done
"$PYTHON_BIN" -m src.aggregate_results --require-all-seeds
