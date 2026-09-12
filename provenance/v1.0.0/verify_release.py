#!/usr/bin/env python3
"""Verify the frozen release using Python's standard library; never train models.

Run from any directory. --selector-replay additionally needs NumPy and replays
the frozen confirmation choices. A manifest match establishes consistency with
this release; it is not an independent authentication of the scientific claims.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


class Checks:
    def __init__(self):
        self.errors = []
        self.passed = 0

    def check(self, condition, message):
        if condition:
            self.passed += 1
        else:
            self.errors.append(message)

    def section(self, name, function):
        before = len(self.errors)
        try:
            function()
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            self.errors.append(f"{name}: {exc}")
        status = "GREEN" if len(self.errors) == before else "RED"
        print(f"{name}: {status}", flush=True)


def verify_manifest(root, checks):
    manifest = read(root / "manifest.json")
    entries = manifest["files"] if isinstance(manifest, dict) else manifest
    checks.check(isinstance(entries, list) and bool(entries), "Manifest has no file list")
    seen = set()
    for item in entries:
        relative = item["path"]
        path = root / relative
        checks.check(relative not in seen, f"Duplicate manifest path: {relative}")
        seen.add(relative)
        if Path(relative).is_absolute() or not path.resolve().is_relative_to(root):
            checks.check(False, f"Manifest path escapes release: {relative}")
            continue
        if not path.is_file():
            checks.check(False, f"Missing release file: {relative}")
            continue
        checks.check(path.stat().st_size == item["bytes"], f"Size changed: {relative}")
        checks.check(sha256(path) == item["sha256"], f"SHA-256 changed: {relative}")
    print(f"  {len(entries):,} manifest entries checked; extra files are allowed.")


def source_digest(source, stage):
    if stage == "confirmation":
        paths = sorted(
            p for p in source.rglob("*")
            if p.is_file() and p.relative_to(source).parts[0]
            in ("lpb", "configs", "development", "tests")
            and p.suffix in (".json", ".py")
        ) + [source / name for name in ("run_experiment.py", "launch.sh", "requirements.txt")]
    else:
        paths = sorted([
            *source.glob("lpb/*.py"), *source.glob("configs/*.json"),
            source / "launch.sh", source / "requirements.txt",
        ])
    # Protocols were made on Linux; use portable '/' keys when checking on Windows.
    return canonical_hash({p.relative_to(source).as_posix(): sha256(p) for p in paths})


def jobs(config, stage):
    if stage == "development":
        families, seeds = config["families"], config["seeds"]
        priors, arms = config["priors"], config["arms"]
    else:
        resolution = stage == "resolution"
        families = config["resolution_families" if resolution else "confirmation_families"]
        seeds = config["resolution_seeds" if resolution else "confirmation_seeds"]
        priors = ["correct", "coefficient"] + ([] if resolution else ["structural"])
        arms = ["none", "smooth"] if resolution else ["none", "iid", "smooth", "response_matched"]
    for family, kind, prior, seed in itertools.product(families, config["kinds"], priors, seeds):
        task = f"{family}__{kind}__{prior}__s{seed}"
        for arm in arms:
            yield dict(task_id=task, family=family, kind=kind, prior=prior,
                       seed=seed, arm=arm, run_id=f"{task}__{arm}")


def verify_records(folder, config, expected_jobs, signature, checks):
    expected_jobs = list(expected_jobs)
    actual = list((folder / "runs").glob("*/record.json"))
    checks.check(len(actual) == len(expected_jobs), f"Unexpected record count in {folder}")
    verified = 0
    for job in expected_jobs:
        run = folder / "runs" / job["run_id"]
        path = run / "record.json"
        if not path.is_file():
            checks.check(False, f"Missing candidate: {job['run_id']}")
            continue
        record = read(path)
        checks.check(all(record.get(key) == value for key, value in job.items()),
                     f"Candidate metadata mismatch: {job['run_id']}")
        checks.check(record.get("completed_steps") == config["steps"],
                     f"Incomplete candidate: {job['run_id']}")
        data = read(folder / "data" / f"{job['family']}_development.json")
        expected = canonical_hash(dict(protocol_hash=signature, job=job, data=data["signature"]))
        checks.check(record.get("signature") == expected, f"Record signature mismatch: {job['run_id']}")
        evaluation = run / "evaluation.npz"
        if evaluation.is_file():
            checks.check(sha256(evaluation) == record.get("evaluation_sha256"),
                         f"Evaluation checksum mismatch: {job['run_id']}")
            verified += 1
        else:
            checks.check(False, f"Missing evaluation: {job['run_id']}")
    print(f"  {folder.name}: {verified}/{len(expected_jobs)} candidate evaluation files checked.")
    return len(expected_jobs)


def verify_experiment(root, stage, checks):
    source = root / "experiments" / stage
    result = root / "results" / stage
    protocol = read(result / "protocol.json")
    config = protocol["config"]
    checks.check(source_digest(source, stage) == protocol["source_hash"],
                 f"{stage} source differs from the recorded protocol")
    if stage == "development":
        payload = dict(schema=protocol["schema"], config=config, source_hash=protocol["source_hash"])
        checks.check(canonical_hash(payload) == protocol["protocol_hash"], "Development protocol hash mismatch")
        total = verify_records(result, config, jobs(config, stage), protocol["protocol_hash"], checks)
        checks.check(total == 864 == protocol["expected_fits"], "Expected 864 development fits")
        manifest = read(result / "selection_manifest.json")
        checks.check(sha256(result / "selection_manifest.json") == read(result / "evaluation_plan.json")["selection_sha256"],
                     "Development selection freeze checksum mismatch")
        checks.check(sha256(result / "selection_inputs.json") == manifest["inputs_sha256"],
                     "Development selection inputs checksum mismatch")
    else:
        payload = dict(config=config, source_hash=protocol["source_hash"])
        checks.check(canonical_hash(payload) == protocol["signature"], "Confirmation protocol hash mismatch")
        total = 0
        for name in ("resolution", "selector"):
            grids = config["resolution_grids"] if name == "resolution" else [config["confirmation_grid"]]
            for grid in grids:
                child = result / name / f"g{grid}"
                subprotocol = read(child / "stage_protocol.json")
                inventory = list(jobs(config, name))
                checks.check(subprotocol["jobs"] == inventory, f"Candidate inventory changed: {name}/g{grid}")
                expected = canonical_hash(dict(parent=protocol["signature"], stage=name, grid=grid))
                checks.check(subprotocol["signature"] == expected, f"Stage signature changed: {name}/g{grid}")
                total += verify_records(child, config, inventory, expected, checks)
                selection = read(child / "selection_manifest.json")
                checks.check(sha256(child / "selection_manifest.json") == read(child / "evaluation_plan.json")["selection_sha256"],
                             f"Selection freeze changed: {name}/g{grid}")
                checks.check(sha256(child / "selection_inputs.json") == selection["inputs_sha256"],
                             f"Selection inputs changed: {name}/g{grid}")
                checks.check(sha256(result / "selector_model.json") == selection["selector_model_sha256"],
                             f"Frozen selector changed: {name}/g{grid}")
        checks.check(total == 720, "Expected 720 confirmation fits")


def verify_development_link(root, checks):
    folder = root / "experiments" / "confirmation" / "development"
    provenance = read(folder / "provenance.json")
    source_bank = root / "results" / "development" / "bank.json"
    filtered_bank = folder / "validation_bank.json"
    checks.check(sha256(source_bank) == provenance["source_bank_sha256"], "Historical source bank hash mismatch")
    checks.check(sha256(filtered_bank) == provenance["included_bank_sha256"], "Filtered development bank hash mismatch")
    source_rows = read(source_bank)
    rows = read(filtered_bank)
    checks.check(len(rows) == provenance["rows"] == 864, "Expected 864 historical validation rows")
    index = {(row["task_id"], row["arm"]): row for row in source_rows}
    allowed = set(provenance["allowed_fields"])
    for row in rows:
        original = index[(row["task_id"], row["arm"])]
        checks.check(row == {key: value for key, value in original.items() if key in allowed},
                     f"Filtered validation row mismatch: {row['task_id']}/{row['arm']}")


def selector_replay(root, checks):
    try:
        import numpy  # noqa: F401
    except ImportError as exc:
        raise ValueError("--selector-replay requires NumPy; omit that flag for a dependency-free integrity check") from exc
    sys.path.insert(0, str(root / "experiments" / "confirmation"))
    from lpb.guarded import decide, fit_selector
    result = root / "results" / "confirmation"
    model = read(result / "selector_model.json")
    child = result / "selector" / "g64"
    inputs = read(child / "selection_inputs.json")
    frozen = read(child / "selection_manifest.json")["decisions"]
    checks.check(decide(model, inputs) == frozen, "Archived model does not exactly replay frozen decisions")
    fitted = fit_selector(read(root / "experiments" / "confirmation" / "development" / "validation_bank.json"))
    checks.check(fitted["selected"] == model["selected"], "Refit selected a different guard configuration")
    replayed = decide(fitted, inputs)
    checks.check(len(replayed) == len(frozen) and all(
        a["task_id"] == b["task_id"] and a["choices"] == b["choices"] for a, b in zip(replayed, frozen)
    ), "Refitted selector policy choices differ from the frozen choices")
    print(f"  {len(frozen)} frozen target decisions replayed; refitted policy choices checked.")
    print("  Coefficients may have harmless floating-point differences across NumPy/BLAS versions.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--selector-replay", action="store_true", help="Also refit/replay the selector using NumPy; no neural training")
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    checks = Checks()
    checks.section("MANIFEST", lambda: verify_manifest(root, checks))
    for stage in ("development", "confirmation"):
        checks.section(stage.upper() + " EVIDENCE", lambda stage=stage: verify_experiment(root, stage, checks))
    checks.section("DEVELOPMENT DATA LINK", lambda: verify_development_link(root, checks))
    if args.selector_replay:
        checks.section("SELECTOR REPLAY", lambda: selector_replay(root, checks))
    status = "RED" if checks.errors else "GREEN"
    print(f"\nRELEASE INTEGRITY: {status} | {checks.passed:,} checks passed; {len(checks.errors)} errors")
    if checks.errors:
        for error in checks.errors[:30]:
            print("  - " + error)
        if len(checks.errors) > 30:
            print(f"  ... {len(checks.errors) - 30} additional errors")
        print("The archived scientific labels are withheld until release integrity passes.")
        return 2
    print("Integrity describes file consistency. Scientific support is reported separately below.")
    print("No model was trained; uncertainty estimates were read from the archived report.")
    report = root / "results" / "confirmation" / "reports" / "report.md"
    print("\nARCHIVED CONFIRMATION REPORT")
    for line in report.read_text(encoding="utf-8").splitlines():
        if not line.startswith("```"):
            print(line)
    print("Historical development labels retain their original exploratory rules; see docs before interpreting them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
