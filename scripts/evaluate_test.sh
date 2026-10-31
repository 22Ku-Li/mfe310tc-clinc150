#!/usr/bin/env bash
set -euo pipefail

if [[ "${FINAL_TEST_CONFIRMATION:-}" != "I_HAVE_FROZEN_ALL_MODELS" ]]; then
  printf 'Refusing final test evaluation. Set FINAL_TEST_CONFIRMATION=I_HAVE_FROZEN_ALL_MODELS only after model/config freeze.\n' >&2
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" -m src.validate_experiment_design
"$PYTHON_BIN" -m src.aggregate_results --require-all-seeds
for experiment in m1_distilbert_ce m2_distilbert_rdrop; do
  for seed in 13 42 87; do
    checkpoint="checkpoints/$experiment/seed_$seed/best"
    if [[ ! -f "$checkpoint/training_state.json" ]]; then
      printf 'Missing frozen checkpoint: %s\n' "$checkpoint" >&2
      exit 2
    fi
  done
done

"$PYTHON_BIN" -m src.train_baseline --config configs/baseline.yaml --split test --allow-test
for checkpoint in checkpoints/m1_distilbert_ce/seed_*/best checkpoints/m2_distilbert_rdrop/seed_*/best; do
  "$PYTHON_BIN" -m src.evaluate_transformer \
    --checkpoint "$checkpoint" \
    --split test \
    --device cuda \
    --allow-test
done
"$PYTHON_BIN" -m src.aggregate_results --require-all-seeds
