#!/usr/bin/env bash
# Run the archived-model calibration follow-up. This launcher never installs Torch.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename -- "${BASH_SOURCE[0]}")"
PROFILE="${1:-plan}"

usage() {
    cat <<'HELP'
Usage: bash launch_calibrated_physics.sh [plan|smoke|pilot|full|report]

Keep physics_calibrated_baseline.zip beside this launcher.
  plan    Inspect the protocol and estimated work; no dependencies installed.
  smoke   Short execution check; NumPy diagnostic fallback if Torch is absent.
  pilot   Small scientific pilot with existing PyTorch, including CPU PyTorch.
  full    Complete calibrated-physics follow-up; no neural training.
  report  Rebuild the report from saved results.

Optional environment settings:
  INSTALL_ROOT      Parent installation folder (default /workspace if writable).
  PYTHON_BIN        Python executable (default python3; no shell arguments).
  BACKEND           torch (default); numpy is diagnostic-only for smoke.
  JOBS              Concurrent calibration workers (default 4).
  THREADS           Threads per worker (default 1).
  TIMING_DEVICE     auto (default), cpu, cuda, or none to skip timings.
  POD_HOURLY_RATE    Your actual hourly Pod rate for cost reporting; optional.
  OUTPUT_DIR        Explicit result folder; default is chosen by the runner.

Example:
  JOBS=4 THREADS=1 bash launch_calibrated_physics.sh pilot

An existing installation is reused only when its manifest matches this ZIP and
all listed files remain unchanged. Saved outputs and .venv are preserved.
For a different release, choose a fresh INSTALL_ROOT. The launcher never removes
existing results and never upgrades or replaces the installed PyTorch package.
HELP
}

if [[ "$PROFILE" == "--help" || "$PROFILE" == "-h" || "$PROFILE" == "help" ]]; then
    usage
    exit 0
fi
if [[ $# -gt 1 ]]; then
    usage >&2
    exit 2
fi
case "$PROFILE" in plan|smoke|pilot|full|report) ;; *) usage >&2; exit 2 ;; esac

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v -- "$PYTHON_BIN" >/dev/null 2>&1; then
    printf 'Python executable unavailable: %s\n' "$PYTHON_BIN" >&2
    exit 2
fi
ARCHIVE="$SCRIPT_DIR/physics_calibrated_baseline.zip"
if [[ ! -f "$ARCHIVE" ]]; then
    printf 'Missing package: %s\nKeep the ZIP beside this launcher.\n' "$ARCHIVE" >&2
    exit 2
fi
if [[ -z "${INSTALL_ROOT:-}" ]]; then
    if [[ -d /workspace && -w /workspace ]]; then
        INSTALL_ROOT=/workspace
    else
        INSTALL_ROOT="$SCRIPT_DIR"
    fi
fi
mkdir -p -- "$INSTALL_ROOT"
INSTALL_ROOT="$(cd -- "$INSTALL_ROOT" && pwd -P)"
PACKAGE_DIR="$INSTALL_ROOT/physics_calibrated_baseline"

# Export before even importing Torch. CUDA visibility is never changed here.
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export JOBS="${JOBS:-4}" THREADS="${THREADS:-1}"
export BACKEND="${BACKEND:-torch}" TIMING_DEVICE="${TIMING_DEVICE:-auto}"
if [[ ! "$JOBS" =~ ^[1-9][0-9]*$ || ! "$THREADS" =~ ^[1-9][0-9]*$ ]]; then
    printf 'JOBS and THREADS must be positive integers.\n' >&2
    exit 2
fi
case "$BACKEND" in torch|numpy) ;; *) printf 'BACKEND must be torch or numpy.\n' >&2; exit 2 ;; esac
if [[ ! "$TIMING_DEVICE" =~ ^(auto|cpu|cuda|none)$ ]]; then
    printf 'TIMING_DEVICE must be auto, cpu, cuda, or none.\n' >&2
    exit 2
fi
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS" NUMEXPR_NUM_THREADS="$THREADS"
export PYTHONUNBUFFERED=1

"$PYTHON_BIN" - "$ARCHIVE" "$PACKAGE_DIR" <<'PY'
import hashlib
import os
import re
import shutil
import stat
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

archive, destination = map(Path, sys.argv[1:])
root_name = 'physics_calibrated_baseline'

def fail(message):
    raise SystemExit('PACKAGE VERIFICATION FAILED: ' + message)

def safe_relative(text):
    if not text or '\\' in text or '\x00' in text:
        fail('invalid archive/manifest path')
    path = PurePosixPath(text)
    if path.is_absolute() or '..' in path.parts or '.' in path.parts:
        fail('unsafe archive/manifest path: ' + text)
    if any(':' in part for part in path.parts):
        fail('unsupported archive/manifest path: ' + text)
    return path

def parse_manifest(raw):
    listed = {}
    try:
        lines = raw.decode('utf-8').splitlines()
    except UnicodeDecodeError:
        fail('manifest must be UTF-8')
    for line in lines:
        if not line.strip():
            continue
        match = re.fullmatch(r'([0-9a-fA-F]{64}) [ *](.+)', line)
        if match is None:
            fail('malformed SHA-256 manifest line')
        digest, filename = match.groups()
        path = safe_relative(filename)
        if str(path) != filename or filename == 'MANIFEST.sha256' or filename in listed:
            fail('duplicate or noncanonical manifest entry: ' + filename)
        listed[filename] = digest.lower()
    if not listed or 'runner.py' not in listed:
        fail('manifest is empty or runner.py is missing')
    return listed

def digest_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def verify_files(folder, listed):
    for filename, expected in listed.items():
        path = folder / filename
        current = folder
        for part in PurePosixPath(filename).parts:
            current = current / part
            if current.is_symlink():
                fail('symlink in installed release: ' + filename)
        if not path.is_file() or digest_file(path) != expected:
            fail('missing or modified release file: ' + filename +
                 '. Preserve this folder and choose a new INSTALL_ROOT for a clean installation.')

try:
    with zipfile.ZipFile(archive) as zipped:
        files, names = {}, set()
        for item in zipped.infolist():
            path = safe_relative(item.filename.rstrip('/'))
            if str(path) != item.filename.rstrip('/') or path.parts[0] != root_name:
                fail('ZIP must contain only the canonical ' + root_name + '/ tree')
            if item.filename in names:
                fail('duplicate ZIP entry: ' + item.filename)
            names.add(item.filename)
            file_mode = item.external_attr >> 16
            if stat.S_ISLNK(file_mode):
                fail('symlinks are not permitted in the package')
            if file_mode and stat.S_IFMT(file_mode) not in (0, stat.S_IFREG, stat.S_IFDIR):
                fail('nonregular ZIP entry: ' + item.filename)
            if item.is_dir():
                continue
            if len(path.parts) < 2:
                fail('invalid package root entry')
            relative = str(PurePosixPath(*path.parts[1:]))
            if relative in files:
                fail('duplicate normalized ZIP path: ' + relative)
            files[relative] = item
        if 'MANIFEST.sha256' not in files:
            fail('MANIFEST.sha256 is missing')
        raw_manifest = zipped.read(files['MANIFEST.sha256'])
        listed = parse_manifest(raw_manifest)
        if set(files) != set(listed) | {'MANIFEST.sha256'}:
            fail('ZIP contents and manifest entries differ')
        # Validate incoming bytes even when an installed release will be reused.
        for filename, expected in listed.items():
            digest = hashlib.sha256()
            with zipped.open(files[filename]) as source:
                for block in iter(lambda: source.read(1024 * 1024), b''):
                    digest.update(block)
            if digest.hexdigest() != expected:
                fail('ZIP checksum mismatch: ' + filename)
        if destination.is_symlink():
            fail('installation destination is a symlink')
        if destination.exists():
            manifest = destination / 'MANIFEST.sha256'
            if not manifest.is_file() or manifest.is_symlink():
                fail('existing installation has no regular manifest. Choose a new INSTALL_ROOT.')
            installed = parse_manifest(manifest.read_bytes())
            if installed != listed:
                fail('ZIP release differs from the existing installation. Saved files were preserved. '
                     'Choose a new INSTALL_ROOT to install this release separately.')
            verify_files(destination, listed)
            print('Verified existing release; saved outputs and environment are preserved.')
        else:
            with tempfile.TemporaryDirectory(prefix='.calibrated_extract_', dir=destination.parent) as staging:
                staged_root = Path(staging) / root_name
                staged_root.mkdir()
                for filename, item in files.items():
                    target = staged_root / filename
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zipped.open(item) as source, target.open('xb') as sink:
                        shutil.copyfileobj(source, sink, length=1024 * 1024)
                    target.chmod(0o755 if ((item.external_attr >> 16) & 0o111) else 0o644)
                verify_files(staged_root, listed)
                os.rename(staged_root, destination)
            print('Installed verified release at ' + str(destination))
except (OSError, ValueError, zipfile.BadZipFile, RuntimeError) as error:
    fail(str(error))
PY

RUN_PYTHON="$PYTHON_BIN"
if [[ "$PROFILE" == "plan" ]]; then
    printf '\nCommands to run after reviewing the plan:\n'
    printf '  bash %q smoke\n  bash %q pilot\n  bash %q full\n  bash %q report\n\n' \
        "$SCRIPT_PATH" "$SCRIPT_PATH" "$SCRIPT_PATH" "$SCRIPT_PATH"
elif [[ "$PROFILE" == "report" ]]; then
    if [[ -x "$PACKAGE_DIR/.venv/bin/python" ]]; then
        RUN_PYTHON="$PACKAGE_DIR/.venv/bin/python"
    fi
else
    if [[ ! -x "$PACKAGE_DIR/.venv/bin/python" ]]; then
        "$PYTHON_BIN" -m venv --system-site-packages "$PACKAGE_DIR/.venv"
    fi
    RUN_PYTHON="$PACKAGE_DIR/.venv/bin/python"
    if [[ "$PROFILE" == "full" || "$PROFILE" == "pilot" ]]; then
        if [[ "$BACKEND" != "torch" ]]; then
            printf 'Pilot/full require BACKEND=torch for scientific comparison with archived records.\n' >&2
            printf 'Use the NumPy backend only for a smoke diagnostic.\n' >&2
            exit 2
        fi
        if ! "$RUN_PYTHON" -c 'import torch; print("Existing PyTorch:", torch.__version__, "| CPU execution is supported")'; then
            printf 'Working PyTorch is required for pilot/full. Use an existing PyTorch environment or template.\n' >&2
            printf 'This launcher does not install or replace Torch. No neural training is performed.\n' >&2
            exit 2
        fi
    elif [[ "$BACKEND" == "torch" ]]; then
        if ! "$RUN_PYTHON" -c 'import torch' >/dev/null 2>&1; then
            export BACKEND=numpy
            printf 'PyTorch unavailable: smoke will use the NumPy diagnostic backend.\n'
            printf 'SCIENTIFIC COMPARISON: NOT_EVALUATED for this diagnostic.\n'
        fi
    fi
    "$RUN_PYTHON" - <<'PY'
import importlib.util
import subprocess
import sys
missing = [name for name in ('numpy', 'scipy') if importlib.util.find_spec(name) is None]
if missing:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', *missing])
PY
fi

cd -- "$PACKAGE_DIR"
printf 'Running profile=%s | backend=%s | jobs=%s | threads=%s | timing=%s\n' \
    "$PROFILE" "$BACKEND" "$JOBS" "$THREADS" "$TIMING_DEVICE"
exec "$RUN_PYTHON" "$PACKAGE_DIR/runner.py" --profile "$PROFILE"
