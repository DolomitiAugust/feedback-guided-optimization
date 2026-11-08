#!/usr/bin/env bash
# Run on Linux with Python 3.10, CUDA toolkit 12.4 and a BF16-capable NVIDIA GPU.
# Creates an isolated environment. No training or cloud purchase is performed.
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$TASK_DIR"
if [[ -e .venv-ppo ]]; then
  echo '.venv-ppo already exists; inspect it before setting up another environment.' >&2
  exit 1
fi
python3.10 -m venv .venv-ppo
source .venv-ppo/bin/activate
python -m pip install --upgrade pip wheel setuptools packaging ninja
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
MAX_JOBS=2 python -m pip install --no-build-isolation flash-attn==2.7.0.post2
python -m pip install -r requirements_ppo.txt
python -m pip check
echo 'Setup complete. Activate .venv-ppo/bin/activate, then follow README.md.'
