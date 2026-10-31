#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"

printf 'Python executable: '
command -v "$PYTHON_BIN"
"$PYTHON_BIN" --version
"$PYTHON_BIN" -m pip --version
nvidia-smi
"$PYTHON_BIN" -c "import torch, transformers; print('torch:', torch.__version__); print('transformers:', transformers.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
