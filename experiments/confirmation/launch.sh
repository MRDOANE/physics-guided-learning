#!/usr/bin/env bash
set -Eeuo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export OMP_NUM_THREADS="${THREADS:-8}"
export MKL_NUM_THREADS="${THREADS:-8}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
action="${1:-plan}"
if (($#)); then shift; fi
if [[ "$action" == plan ]]; then
  exec "$PYTHON_BIN" run_experiment.py plan "$@"
fi
if ! "$PYTHON_BIN" -c 'import torch,numpy; assert tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) >= (2,8)' >/dev/null 2>&1; then
  echo '[RED] ENVIRONMENT: Python needs NumPy and PyTorch >=2.8. Select a RunPod PyTorch GPU template. This launcher preserves your installed Torch.' >&2
  exit 2
fi
exec "$PYTHON_BIN" -u run_experiment.py "$action" --device "${DEVICE:-cuda}" --threads "${THREADS:-8}" "$@"
