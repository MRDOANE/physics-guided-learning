#!/usr/bin/env bash
# Run inside the extracted physics_family_selector package.
set -Eeuo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

MODE="${1:-full}"
if [[ $# -gt 0 ]]; then shift; fi
case "$MODE" in
    smoke|pilot|full) PROFILE="$MODE" ;;
    report) PROFILE="${PROFILE:-full}" ;;
    plan) PROFILE="${1:-full}"; if [[ $# -gt 0 ]]; then shift; fi ;;
    *) echo 'Usage: bash launch.sh {smoke|pilot|full|report|plan [profile]} [CLI arguments]' >&2; exit 2 ;;
esac
case "$PROFILE" in smoke|pilot|full) ;; *) echo "Unknown profile: $PROFILE" >&2; exit 2 ;; esac
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit('Python 3.10 or newer is required. Choose a recent RunPod PyTorch template.')
PY

if [[ ! -d .venv ]]; then
    "$PYTHON_BIN" -m venv --system-site-packages .venv
fi
if [[ ! -x .venv/bin/python ]]; then
    echo 'The existing .venv is incomplete. Rename it and rerun this launcher.' >&2
    exit 2
fi
PYTHON="$PWD/.venv/bin/python"
"$PYTHON" - <<'PY'
try:
    import torch
except ImportError:
    raise SystemExit('PyTorch is missing. Use a RunPod PyTorch template; this launcher preserves the template torch/CUDA installation.')
print('Python/PyTorch environment ready:', torch.__version__, flush=True)
PY
if ! "$PYTHON" - <<'PY'
try:
    import numpy as np
    from numpy.lib import NumpyVersion
    valid = NumpyVersion('1.26.0') <= NumpyVersion(np.__version__) < NumpyVersion('3.0.0')
except ImportError:
    valid = False
raise SystemExit(0 if valid else 1)
PY
then
    "$PYTHON" -m pip install --disable-pip-version-check -r requirements.txt
fi

THREADS="${THREADS:-8}"
if [[ ! "$THREADS" =~ ^[1-9][0-9]*$ ]]; then echo 'THREADS must be a positive integer.' >&2; exit 2; fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$THREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$THREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$THREADS}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONUNBUFFERED=1

if [[ "$MODE" == plan ]]; then
    exec "$PYTHON" -m lpb.cli plan --profile "$PROFILE" "$@"
fi
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/outputs/$PROFILE}"
OUTPUT_DIR="$("$PYTHON" -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).expanduser().resolve())' "$OUTPUT_DIR")"
if [[ "$MODE" == report ]]; then
    exec "$PYTHON" -m lpb.cli report --output "$OUTPUT_DIR" --color auto "$@"
fi
if [[ "$PROFILE" == full ]]; then DEFAULT_DEVICE=cuda; else DEFAULT_DEVICE=auto; fi
DEVICE="${DEVICE:-$DEFAULT_DEVICE}"
case "$DEVICE" in auto|cpu|cuda) ;; *) echo 'DEVICE must be auto, cpu, or cuda.' >&2; exit 2 ;; esac
"$PYTHON" - "$DEVICE" <<'PY'
import sys, torch
if sys.argv[1] == 'cuda' and not torch.cuda.is_available():
    raise SystemExit('CUDA is unavailable. Use a RunPod PyTorch GPU template; this launcher does not replace torch. For a CPU check, set DEVICE=cpu.')
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0), flush=True)
PY
mkdir -p -- "$OUTPUT_DIR"
if command -v flock >/dev/null 2>&1; then
    exec 9>"$OUTPUT_DIR/.launcher.lock"
    flock -n 9 || { echo "Another launcher is using $OUTPUT_DIR" >&2; exit 3; }
fi
echo "Profile: $PROFILE | Device: $DEVICE | CPU threads: $THREADS"
echo "Results: $OUTPUT_DIR"
exec "$PYTHON" -m lpb.cli run --profile "$PROFILE" --output "$OUTPUT_DIR" --device "$DEVICE" --threads "$THREADS" "$@"
