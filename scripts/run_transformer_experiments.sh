#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8

"$PYTHON_BIN" -m src.download_data
"$PYTHON_BIN" -m src.data
"$PYTHON_BIN" -m src.validate_experiment_design
"$PYTHON_BIN" -m src.prepare_model --config configs/distilbert.yaml

for config in configs/distilbert.yaml configs/rdrop.yaml; do
  if [[ "$config" == "configs/distilbert.yaml" ]]; then
    experiment="m1_distilbert_ce"
  else
    experiment="m2_distilbert_rdrop"
  fi
  for seed in 13 42 87; do
    mkdir -p "results/$experiment/seed_$seed"
    "$PYTHON_BIN" -m src.train_transformer \
      --config "$config" \
      --seed "$seed" \
      --device cuda \
      2>&1 | tee "results/$experiment/seed_$seed/gpu_console.log"
  done
done

"$PYTHON_BIN" -m src.aggregate_results --require-all-seeds
