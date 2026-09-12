#!/usr/bin/env bash
set -euo pipefail
audit_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [ "$#" -lt 1 ]; then
  echo "Usage: bash launch_timing_no_training.sh /path/to/physics-guided-learning [additional options]"
  echo "Uses the existing Python/PyTorch environment. No model training or package installation."
  exit 2
fi
audit_repo_path="$1"
shift
exec "${PYTHON:-python3}" "$audit_script_dir/benchmark_inference_no_training.py" --repository "$audit_repo_path" "$@"
