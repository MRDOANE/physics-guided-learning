#!/usr/bin/env python3
"""Recompute archived statistics on a new copy; never train or change evidence.

Uses the original experiment's report code and requires NumPy and PyTorch.
The CPU build of PyTorch is sufficient. Report replay does not require a GPU.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("confirmation", "development"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    missing = [name for name in ("numpy", "torch") if importlib.util.find_spec(name) is None]
    if missing:
        print("REPORT REPLAY: NOT RUN | missing " + ", ".join(missing), file=sys.stderr)
        print("Use a Python environment with NumPy and PyTorch. A CPU PyTorch build is sufficient; no GPU or training is used.", file=sys.stderr)
        print("For a check that only needs Python, run scripts/verify_release.py instead.", file=sys.stderr)
        return 2
    if args.stage == "confirmation" and importlib.util.find_spec("fcntl") is None:
        print("REPORT REPLAY: NOT RUN | original confirmation imports require Linux/macOS/WSL (fcntl).", file=sys.stderr)
        return 2
    source = root / "experiments" / args.stage
    evidence = root / "results" / args.stage
    if not source.is_dir() or not (evidence / "protocol.json").is_file():
        print("REPORT REPLAY: NOT RUN | missing experiment source or archived evidence.", file=sys.stderr)
        return 2
    parent = root / "outputs" / "report_replay"
    destination = parent / args.stage
    suffix = 2
    while destination.exists():
        destination = parent / f"{args.stage}_{suffix}"
        suffix += 1
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(evidence, destination)
        # The original report reads development provenance from this location.
        shutil.copytree(source, destination / "source", ignore=shutil.ignore_patterns("__pycache__", ".venv", "outputs"))
    except OSError as exc:
        print(f"REPORT REPLAY: NOT RUN | could not make a fresh evidence copy: {exc}", file=sys.stderr)
        return 2
    print(f"Report replay copy: {destination}", flush=True)
    print("Archived results are preserved. The following command recomputes statistics; no model is trained.", flush=True)
    if args.stage == "confirmation":
        program = "from pathlib import Path; import sys; from lpb.confirmation_report import report; sys.exit(report(Path(sys.argv[1]))['exit_code'])"
        command = [sys.executable, "-c", program, str(destination)]
    else:
        command = [sys.executable, "-m", "lpb.cli", "report", "--output", str(destination), "--color", "auto"]
    completed = subprocess.run(command, cwd=source, check=False)
    if completed.returncode:
        print(f"REPORT REPLAY: RED | report process returned {completed.returncode}; the original archive is unchanged.", file=sys.stderr)
        return completed.returncode
    try:
        result = json.loads((destination / "reports" / "report.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"REPORT REPLAY: RED | could not read the recomputed report: {exc}", file=sys.stderr)
        return 2
    completion = result.get("completion", {})
    if result.get("errors") or completion.get("label", completion.get("status")) == "RED":
        print("REPORT REPLAY: RED | the recomputed report identifies an integrity or completion error.", file=sys.stderr)
        print(f"Read {destination / 'reports' / 'report.json'}", file=sys.stderr)
        return 2
    print(f"REPORT REPLAY: GREEN | recomputed reports saved in {destination / 'reports'}")
    print("This GREEN label means replay completed. Scientific support remains comparison-specific.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
