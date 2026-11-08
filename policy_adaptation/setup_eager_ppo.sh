#!/usr/bin/env bash
# Fresh Linux/Python 3.10 environment for the measured single-GPU eager variant.
# This installs dependencies and checked patches; it does not start training.
set -euo pipefail
cd "$(dirname "$0")"
python - <<'PY'
import sys, shutil
assert sys.platform == "linux" and sys.version_info[:2] == (3, 10)
assert shutil.which("nvcc"), "A compatible CUDA toolkit is required."
PY
export MAX_JOBS=2 NVCC_THREADS=2 DS_BUILD_OPS=0
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export PIP_NO_INPUT=1 PIP_DISABLE_PIP_VERSION_CHECK=1
INDEX_URL="${RESEARCH_PYPI_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
python -m pip install --index-url "$INDEX_URL" setuptools==80.9.0 wheel==0.45.1 packaging==25.0 ninja==1.11.1.4
python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1
python -m pip install --index-url "$INDEX_URL" --no-deps --no-build-isolation \
  'openrlhf @ git+https://github.com/OpenRLHF/OpenRLHF.git@bfa85f6fa8cd6214707f93883fac5d08334f4f06'
python -m pip install --index-url "$INDEX_URL" --no-build-isolation --prefer-binary \
  -c results/constraints_eager_retry.txt \
  -r results/requirements_eager_retry.txt
python - <<'PY'
from pathlib import Path
from optional_flash_imports import apply_optional_flash_patch
print(apply_optional_flash_patch(Path("logs")))
PY
python patch_openrlhf.py
python - <<'PY'
import subprocess, sys
import torch
import openrlhf.cli.train_ppo
from deepspeed.ops.adam import FusedAdam
assert torch.cuda.is_available() and torch.__version__ == "2.5.1+cu124"
p = torch.nn.Parameter(torch.tensor([1.0], device="cuda"))
optimizer = FusedAdam([p], lr=0.01)
p.square().sum().backward()
optimizer.step()
torch.cuda.synchronize()
assert torch.isfinite(p).all().item() and p.item() != 1.0
result = subprocess.run([sys.executable, "-m", "pip", "check"], text=True, capture_output=True)
lines = [line.strip() for line in (result.stdout + result.stderr).splitlines() if line.strip()]
assert result.returncode == 1 and lines and all(
    line.lower().startswith("openrlhf ") and "requires flash-attn, which is not installed" in line.lower()
    for line in lines), "Unexpected dependency conflict: " + str(lines)
print("Eager environment and actual CUDA optimizer checked. Unused Flash declaration remains visible.")
print("This installer has not run the policy training experiment.")
PY
