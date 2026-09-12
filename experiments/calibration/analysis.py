"""Paired, exploratory reporting for the calibrated-physics extension.

This module fits no models. It reads frozen choices, verifies local result hashes,
and combines compatible evaluations with archived neural trajectory scores.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


CAP = 1_000_000.0
LABELS = {
    "GREEN": "candidate has lower error; pointwise interval is above zero",
    "RED": "candidate has higher error; pointwise interval is below zero",
    "YELLOW": "pointwise interval includes zero",
    "BLUE": "the compared trajectory scores are identical",
    "NOT_EVALUATED": "diagnostic profile; no scientific interpretation",
}


def _read(path):
    return json.loads(Path(path).read_text())


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_jsonable(value), indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def _csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    if not fields:
        fields = ["status"]
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v
                             for k, v in _jsonable(row).items()})


def _scores(curve, horizon, stage):
    curve = np.asarray(curve, dtype=np.float64)
    if curve.ndim != 2 or curve.shape[1] < horizon or curve.shape[0] < 1:
        raise ValueError("Evaluation curve has an incompatible shape")
    x = curve[:, :horizon]
    if np.any(x[np.isfinite(x)] < 0):
        raise ValueError("Squared-error curves contain negative values")
    with np.errstate(over="ignore", invalid="ignore"):
        raw = x.mean(axis=1)
    if stage == "development":
        result = np.minimum(np.nan_to_num(raw, nan=CAP, posinf=CAP, neginf=CAP), CAP)
        affected = ~np.isfinite(raw) | (raw > CAP)
    else:
        result = np.clip(np.nan_to_num(x, nan=CAP, posinf=CAP, neginf=CAP), 0, CAP).mean(axis=1)
        affected = np.any(~np.isfinite(x) | (x > CAP), axis=1)
    return result, {
        "failed_trajectories": int(np.any(~np.isfinite(x), axis=1).sum()),
        "cap_affected_trajectories": int(affected.sum()),
        "nonfinite_steps": int((~np.isfinite(x)).sum()),
        "raw_mean_nmse": float(raw.mean()) if np.isfinite(raw).all() else None,
    }


def _archive(assets):
    metadata = _read(assets / "record_metadata.json")
    index = _read(assets / "score_index.json")
    vectors = {}
    with np.load(assets / "paired_trajectory_scores.npz", allow_pickle=False) as archive:
        for entry in index:
            if entry["category"] == "neural":
                vectors[tuple(entry["key"])] = np.asarray(archive[entry["array"]], dtype=np.float64).copy()
    return metadata, vectors


def _draws(matrix, trajectory_weights, seed_weights):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape[0] == 1:
        return trajectory_weights @ matrix[0]
    if matrix.shape[0] != seed_weights.shape[1]:
        raise ValueError("Training-seed pairing changed between neural candidates")
    return np.sum((trajectory_weights @ matrix.T) * seed_weights, axis=1)


def _same_scores(a, b):
    if a.shape[1] != b.shape[1]:
        return False
    if a.shape[0] == b.shape[0]:
        return np.array_equal(a, b)
    if a.shape[0] == 1:
        return np.array_equal(np.broadcast_to(a, b.shape), b)
    if b.shape[0] == 1:
        return np.array_equal(a, np.broadcast_to(b, a.shape))
    return False


def _contrast(base, candidate, baseline_matrix, candidate_matrix, baseline_draws,
              candidate_draws, baseline_name, candidate_name, scientific):
    delta = baseline_draws - candidate_draws
    lower, upper = np.quantile(delta, [.025, .975])
    same = _same_scores(baseline_matrix, candidate_matrix)
    if same:
        difference = lower = upper = 0.0
    else:
        difference = float(baseline_matrix.mean() - candidate_matrix.mean())
    if not scientific:
        label = "NOT_EVALUATED"
    elif same:
        label = "BLUE"
    elif lower > 0:
        label = "GREEN"
    elif upper < 0:
        label = "RED"
    else:
        label = "YELLOW"
    point_base = float(baseline_matrix.mean())
    relative = 100 * difference / point_base if point_base > 0 else None
    valid_ratio = baseline_draws > 0
    if valid_ratio.all():
        relative_draws = 100 * delta / baseline_draws
        relative_low, relative_high = np.quantile(relative_draws, [.025, .975])
        if same:
            relative_low = relative_high = 0.0
    else:
        relative_low = relative_high = None
    return {
        **base, "candidate": candidate_name, "baseline": baseline_name,
        "candidate_mean_nmse": float(candidate_matrix.mean()), "baseline_mean_nmse": point_base,
        "difference_nmse_baseline_minus_candidate": difference,
        "difference_ci95_low": float(lower), "difference_ci95_high": float(upper),
        "mse_reduction_percent": relative, "reduction_ci95_low": relative_low,
        "reduction_ci95_high": relative_high, "undefined_ratio_draws": int((~valid_ratio).sum()),
        "label": label, "interpretation": LABELS[label],
        "inference_scope": "exploratory pointwise 95% paired bootstrap; no multiplicity adjustment; no minimum effect requirement",
    }


def _format(value, digits=4):
    if value is None:
        return "—"
    if isinstance(value, (int, np.integer)):
        return str(value)
    return f"{float(value):.{digits}g}"


def _method(decision):
    value = str(decision.get("selected_method", "")).lower()
    if value in ("calibrated", "calibrated_physics", "physics_calibrated"):
        return "calibrated"
    if value in ("uncalibrated", "uncalibrated_physics", "physics_uncalibrated", "mechanistic", "physics"):
        return "uncalibrated"
    if value in ("neural", "model", "nn", "transformer", "looped", "fno"):
        return "neural"
    raise ValueError(f"Unknown frozen selected_method {value!r}")


def _audit_decisions(job, calibration, decisions, metadata):
    """Reconstruct the frozen rule using validation records only."""
    records = [r for r in metadata if r["stage"] == job["stage"]
               and r["family"] == job["family"] and r["prior"] == job["prior"]]
    seeds = sorted({int(r["seed"]) for r in records})
    if not seeds or len(decisions) != len(seeds) or sorted(int(d["seed"]) for d in decisions) != seeds:
        raise ValueError("Frozen validation decisions do not match archived training seeds")
    for decision in decisions:
        group = sorted((r for r in records if int(r["seed"]) == int(decision["seed"])),
                       key=lambda r: (float(r["final_val"]), r["kind"], r["arm"]))
        best = group[0]
        if decision["neural_kind"] != best["kind"] or decision["neural_arm"] != best["arm"]:
            raise ValueError("Frozen best-neural choice differs from the declared validation/tie rule")
        if float(decision["neural_validation_nmse"]) != float(best["final_val"]):
            raise ValueError("Frozen neural validation score differs from the archive")
        choices = [("uncalibrated", float(calibration["identity_validation_nmse"])),
                   ("calibrated", float(calibration["validation_nmse"])),
                   ("neural", float(best["final_val"]))]
        if not all(math.isfinite(score) for _, score in choices):
            raise ValueError("Frozen model-choice inputs contain nonfinite validation scores")
        chosen_method, chosen_score = min(choices, key=lambda item: item[1])
        if _method(decision) != chosen_method or float(decision["validation_nmse"]) != chosen_score:
            raise ValueError("Frozen selected route differs from the minimum validation loss or declared tie rule")
        expected_kind = best["kind"] if chosen_method == "neural" else None
        expected_arm = best["arm"] if chosen_method == "neural" else None
        if decision.get("selected_kind") != expected_kind or decision.get("selected_arm") != expected_arm:
            raise ValueError("Frozen selected neural identity does not match the selected route")
    return seeds, records


def _timing(output):
    records, rows, issues = [], [], []
    folder = output / "timing"
    for path in sorted(folder.glob("*.json")) if folder.exists() else []:
        try:
            item = _read(path)
            records.append({"path": str(path.relative_to(output)), "sha256": _sha(path), "record": item})
            environment = item.get("environment", {})
            for row in item.get("rows", []):
                rows.append({"source_file": str(path.relative_to(output)),
                             "device": environment.get("device"), "gpu": environment.get("gpu"),
                             "threads": environment.get("threads"), **row})
        except (OSError, ValueError, TypeError) as exc:
            issues.append(f"Timing report {path.name}: {type(exc).__name__}: {exc}")
    return records, rows, issues


def analyze(output: Path, assets: Path, bootstrap: int = 5000) -> dict:
    """Verify completed settings and write a report without fitting any model.

    Missing settings yield a partial completion report. Hash mismatches withhold
    the affected setting. Archive incompatibility withholds its neural comparisons
    and leaves independent calibrated-versus-uncalibrated evaluations available.
    """
    output, assets = Path(output), Path(assets)
    bootstrap = int(bootstrap)
    if bootstrap < 100:
        raise ValueError("At least 100 bootstrap draws are required for a diagnostic interval")
    reports = output / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    protocol = _read(output / "protocol.json")
    freeze = _read(output / "freeze.json") if (output / "freeze.json").exists() else {}
    execution = _read(output / "execution.json") if (output / "execution.json").exists() else {}
    profile = protocol.get("profile", "unknown")
    scientific = profile in ("full", "pilot") and "numpy" not in str(protocol.get("backend", "")).lower()
    primary = 16 if profile == "smoke" else 64
    horizons = (16, 32) if profile == "smoke" else (64, 96)
    archive_metadata = archive_vectors = None
    absolute, comparisons, diagnostics, job_reports = [], [], [], []
    errors, pending, notices = [], [], []
    requested = protocol.get("jobs", [])
    verified = 0
    freeze_protocol_valid = not freeze or freeze.get("protocol_sha256") == _sha(output / "protocol.json")
    if not freeze_protocol_valid:
        errors.append({"job_id": "ALL", "reason": "Frozen protocol hash does not match protocol.json"})
    for job in requested:
        job_id = job["id"]
        folder = output / "jobs" / job_id
        calibration_path, evaluation_path = folder / "calibration.json", folder / "evaluation.npz"
        metadata_path = folder / "evaluation.json"
        missing = [p.name for p in (calibration_path, evaluation_path, metadata_path) if not p.exists()]
        if missing or job_id not in freeze.get("calibrations_sha256", {}):
            pending.append({"job_id": job_id, "reason": "Missing files or frozen calibration", "missing": missing})
            continue
        if not freeze_protocol_valid:
            continue
        try:
            calibration_sha, freeze_sha = _sha(calibration_path), _sha(output / "freeze.json")
            if calibration_sha != freeze["calibrations_sha256"][job_id]:
                raise ValueError("Calibration hash does not match the frozen parameter file")
            calibration, evaluation = _read(calibration_path), _read(metadata_path)
            if evaluation.get("evaluation_sha256") != _sha(evaluation_path):
                raise ValueError("Evaluation-array checksum mismatch")
            if evaluation.get("calibration_sha256") != calibration_sha:
                raise ValueError("Evaluation metadata do not match the frozen calibration hash")
            if evaluation.get("freeze_sha256") != freeze_sha:
                raise ValueError("Evaluation metadata do not match the selection-freeze hash")
            if calibration.get("status") != "FITTED_AND_VALIDATED":
                raise ValueError(f"Calibration status {calibration.get('status')!r} is incomplete")
            factors = np.asarray(calibration.get("selected_multipliers", calibration.get("multipliers")), dtype=float)
            if factors.shape != (3,) or not np.isfinite(factors).all() or np.any(factors <= 0):
                raise ValueError("Frozen calibration does not contain three finite positive multipliers")
            with np.load(evaluation_path, allow_pickle=False) as src:
                curves = {key: np.asarray(src[key], dtype=float).copy()
                          for key in (f"{split}_{method}" for split in ("test", "ood")
                                      for method in ("calibrated", "uncalibrated", "persistence"))}
            shapes = {value.shape for value in curves.values()}
            expected_count = 8 if profile == "smoke" else 64
            if len(shapes) != 1 or any(len(shape) != 2 or shape[0] != expected_count or shape[1] < max(horizons) for shape in shapes):
                raise ValueError("Evaluation-array counts or forecast horizons differ")
            for split in ("test", "ood"):
                for method in ("calibrated", "uncalibrated", "persistence"):
                    _scores(curves[f"{split}_{method}"], max(horizons), job["stage"])
            # Record metadata contain validation scores, not held-out errors.
            # Validate the freeze even when test-array compatibility fails.
            if archive_metadata is None:
                archive_metadata = _read(assets / "record_metadata.json")
            decisions = freeze.get("decisions", {}).get(job_id, [])
            seeds, records = _audit_decisions(job, calibration, decisions, archive_metadata)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append({"job_id": job_id, "reason": f"{type(exc).__name__}: {exc}"})
            continue
        verified += 1
        compatible = bool(evaluation.get("archive_compatible", False)) and profile != "smoke"
        neural_issue = None
        if compatible:
            try:
                if archive_vectors is None:
                    archive_metadata, archive_vectors = _archive(assets)
                for split in ("test", "ood"):
                    for horizon in horizons:
                        for r in records:
                            key = (job["stage"], job["family"], r["kind"], job["prior"], int(r["seed"]), r["arm"], split, horizon)
                            vector = archive_vectors[key]
                            if vector.shape != (curves[f"{split}_calibrated"].shape[0],) or not np.isfinite(vector).all() or np.any(vector < 0):
                                raise ValueError("Archived neural trajectory scores are incompatible")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                compatible = False
                neural_issue = f"{type(exc).__name__}: {exc}"
                notices.append(f"{job_id}: neural comparisons withheld: {neural_issue}")
        elif profile != "smoke":
            notices.append(f"{job_id}: archive compatibility was not established; neural comparisons withheld")
        decisions = sorted(decisions, key=lambda d: int(d["seed"])) if compatible else []
        if compatible and evaluation.get("compatibility", {}).get("status") == "EXACT_DATA_HASH_REFERENCE_ROUNDING_DIFFERS":
            notices.append(f"{job_id}: exact development/test data hashes establish shared targets; CPU/GPU physical-reference rounding differences remain recorded")
        job_report = {**job, "verified": True, "archive_compatible": compatible,
                      "compatibility": evaluation.get("compatibility"), "neural_comparison_issue": neural_issue,
                      "selected_multipliers": factors.tolist(), "selected_candidate": calibration["selected_candidate"],
                      "validation_nmse": calibration["validation_nmse"],
                      "identity_validation_nmse": calibration.get("identity_validation_nmse"),
                      "fit_seconds": calibration.get("fit_seconds"),
                      "calibration_total_seconds": calibration.get("total_seconds"),
                      "calibration_settings": calibration.get("settings", {}),
                      "calibration_diagnostics": {k: calibration.get(k) for k in
                          ("candidate_count", "valid_candidate_count", "selected_boundary_hits", "residual_calls", "invalid_residual_calls")},
                      "candidate_records": calibration.get("candidate_records", []),
                      "frozen_decisions": decisions}
        job_reports.append(job_report)
        for split in ("test", "ood"):
            count = curves[f"{split}_calibrated"].shape[0]
            seed_value = int(hashlib.sha256(f"20260913:{job_id}:{split}".encode()).hexdigest()[:16], 16)
            rng = np.random.default_rng(seed_value)
            tw = rng.multinomial(count, np.full(count, 1 / count), size=bootstrap) / count
            ns = len(seeds) if compatible else 1
            sw = rng.multinomial(ns, np.full(ns, 1 / ns), size=bootstrap) / ns
            for horizon in horizons:
                base = {"job_id": job_id, "stage": job["stage"], "family": job["family"],
                        "prior": job["prior"], "grid": job["grid"], "split": split, "horizon": horizon,
                        "n_trajectories": count, "primary_horizon": horizon == primary}
                matrices, details = {}, {}
                for method in ("calibrated", "uncalibrated", "persistence"):
                    values, diagnostic = _scores(curves[f"{split}_{method}"], horizon, job["stage"])
                    matrices[method] = values[None]
                    details[method] = {"kind": f"physics_{method}" if method != "persistence" else "persistence",
                                       "arm": "", "source": "new_frozen_physics_evaluation", **diagnostic,
                                       "calibration_seconds_current_machine": calibration.get("total_seconds") if method == "calibrated" else 0.0}
                    diagnostics.append({**base, "method": method, **diagnostic})
                if compatible:
                    groups = sorted({(r["kind"], r["arm"]) for r in records})
                    by_key = {(int(r["seed"]), r["kind"], r["arm"]): r for r in records}
                    def neural_vector(seed, kind, arm):
                        return archive_vectors[(job["stage"], job["family"], kind, job["prior"], int(seed), arm, split, horizon)]
                    for kind, arm in groups:
                        name = f"neural:{kind}:{arm}"
                        matrices[name] = np.stack([neural_vector(seed, kind, arm) for seed in seeds])
                        rr = [by_key[(seed, kind, arm)] for seed in seeds]
                        route = [float(r["train_seconds"]) + float(r["setup_seconds"]) +
                                 (0 if arm == "none" else float(r["bank_setup_seconds"])) for r in rr]
                        details[name] = {"kind": kind, "arm": arm, "source": "archived_neural_scores",
                                         "archived_mean_training_validation_seconds": float(np.mean([r["train_seconds"] for r in rr])),
                                         "archived_mean_fixed_choice_route_seconds": float(np.mean(route)),
                                         "archived_median_inference_ms_per_step": float(np.median([r["inference_ms_per_step"] for r in rr])),
                                         "archived_parameter_count": rr[0].get("parameter_count"),
                                         "archived_gpu": rr[0].get("stack", {}).get("gpu"),
                                         "failed_trajectories": None, "cap_affected_trajectories": None,
                                         "cost_scope": "historical execution; no matched-hardware comparison with current calibration times"}
                    matrices["validation_best_neural"] = np.stack([neural_vector(d["seed"], d["neural_kind"], d["neural_arm"]) for d in decisions])
                    route = []
                    for decision in decisions:
                        method = _method(decision)
                        route.append(neural_vector(decision["seed"], decision["selected_kind"], decision["selected_arm"])
                                     if method == "neural" else matrices[method][0])
                    matrices["validation_selected_route"] = np.stack(route)
                    for method in ("validation_best_neural", "validation_selected_route"):
                        details[method] = {"kind": method, "arm": "", "source": "frozen_validation_selection",
                                           "failed_trajectories": None, "cap_affected_trajectories": None}
                draws = {name: _draws(matrix, tw, sw) for name, matrix in matrices.items()}
                for name, matrix in matrices.items():
                    lower, upper = np.quantile(draws[name], [.025, .975])
                    absolute.append({**base, "method": name, "mean_nmse": float(matrix.mean()),
                                     "ci95_low": float(lower), "ci95_high": float(upper),
                                     "normalized_rmse": float(np.sqrt(matrix.mean())),
                                     "n_training_seeds": matrix.shape[0] if name.startswith("neural:") or name.startswith("validation_") else 0,
                                     **details[name]})
                pairs = [("uncalibrated", "calibrated")]
                if compatible:
                    pairs.extend([("validation_best_neural", "calibrated"),
                                  ("validation_best_neural", "validation_selected_route"),
                                  ("calibrated", "validation_selected_route")])
                for baseline, candidate in pairs:
                    comparisons.append(_contrast(base, candidate, matrices[baseline], matrices[candidate],
                                                  draws[baseline], draws[candidate], baseline, candidate, scientific))

    timing_records, timing_rows, timing_issues = _timing(output)
    notices.extend(timing_issues)
    completion = "RED" if errors else "GREEN" if verified == len(requested) and requested else "YELLOW"
    scientific_status = "EXPLORATORY_COMPLETED_SETTINGS" if scientific and verified else "NOT_EVALUATED"
    elapsed = execution.get("elapsed_seconds")
    rate = execution.get("pod_hourly_rate")
    estimated_cost = float(elapsed) / 3600 * float(rate) if elapsed is not None and rate is not None else None
    report = {
        "schema_version": 1, "completion": completion, "verified_settings": verified,
        "requested_settings": len(requested), "profile": profile, "backend": protocol.get("backend"),
        "scientific_evidence": scientific_status, "primary_horizon": primary,
        "statistical_scope": "Exploratory pointwise 95% paired bootstrap intervals; no multiplicity adjustment or global acceptance rule.",
        "uncertainty_scope": "Shared test trajectories and, for neural options, archived training seeds are resampled. The fitted physical solver and training dataset are held fixed; intervals omit calibration/training-data uncertainty.",
        "selection_scope": "Frozen validation choices use archived neural final_val and the new 16-step physics validation scores. Test errors are used only for evaluation. Prior test inspection makes this a post hoc study extension.",
        "scoring": "development caps each trajectory mean at 1e6; confirmation caps each per-step error at 1e6 before averaging. Nonfinite errors are assigned the cap; failed trajectories are retained.",
        "labels": LABELS, "minimum_effect_requirement": None, "minimum_successful_system_count": None,
        "bootstrap_replicates": bootstrap, "neural_training_performed": False,
        "jobs": job_reports, "absolute_accuracy": absolute, "comparisons": comparisons,
        "numerical_diagnostics": diagnostics, "integrity_errors": errors, "pending_settings": pending,
        "notices": notices, "execution": execution, "estimated_pod_compute_cost": estimated_cost,
        "cost_scope": "Current elapsed time times supplied hourly rate; storage and idle time excluded. Historical neural costs describe their recorded machine. Fresh matched timings are separate initialized-architecture benchmarks.",
        "fresh_timing_records": timing_records,
    }
    _write_json(reports / "report.json", report)
    _csv(reports / "absolute_accuracy.csv", absolute)
    _csv(reports / "comparisons.csv", comparisons)
    _csv(reports / "numerical_diagnostics.csv", diagnostics)
    _csv(reports / "matched_timing.csv", timing_rows)
    lines = ["# Calibrated physics and neural forecast comparison", "",
             f"Completion: **{completion}** — {verified}/{len(requested)} settings verified.",
             f"Scientific evidence: **{scientific_status}**. Profile: `{profile}`; backend: `{protocol.get('backend')}`.", "",
             "This extension tests whether fitting a small physical-parameter correction to the existing training trajectories changes the choice between a solver and a neural forecaster. Neural weights are reused through their archived prediction errors. No neural model is retrained.", "",
             "Three positive multipliers are fitted per equation/prior/grid setting. They are shared across trajectories and remain fixed for both test distributions. Structural omissions remain in the equations. The unchanged solver is included in validation selection.", "",
             "## Reading the evidence", "",
             "GREEN indicates lower candidate error with a pointwise interval above zero; RED indicates higher error; YELLOW means the interval includes zero; BLUE means identical trajectory scores. Diagnostic profiles use NOT_EVALUATED. Positive differences and positive percentages favor the candidate.", "",
             "There is no minimum required improvement and no required number of successful equations. Labels describe individual exploratory comparisons. They do not decide whether the paper or a general hypothesis succeeds.", "",
             report["statistical_scope"], report["uncertainty_scope"], "",
             "## Primary in-distribution accuracy", "",
             f"Errors average the first {primary} forecast steps. NMSE uses the original training-derived channel scales. Lower values are better. Equations remain separate.", "",
             "| Setting | Uncalibrated physics | Calibrated physics | Validation-best neural | Validation-selected route |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for job in job_reports:
        rr = {r["method"]: r for r in absolute if r["job_id"] == job["id"] and r["split"] == "test" and r["horizon"] == primary}
        def val(method):
            return _format(rr.get(method, {}).get("mean_nmse"))
        lines.append(f"| {job['family']} / {job['prior']} / g{job['grid']} | {val('uncalibrated')} | {val('calibrated')} | {val('validation_best_neural')} | {val('validation_selected_route')} |")
    lines += ["", "Every neural architecture and augmentation option, both distributions, both horizons, pointwise intervals, and archived cost measurements appear in `absolute_accuracy.csv`.", "",
              "## Calibrated-physics contrasts", "",
              "| Setting | Distribution / steps | Baseline | Label | MSE reduction, % [95% interval] |",
              "| --- | --- | --- | --- | ---: |"]
    for row in comparisons:
        if row["candidate"] != "calibrated":
            continue
        relative = f"{_format(row['mse_reduction_percent'])} [{_format(row['reduction_ci95_low'])}, {_format(row['reduction_ci95_high'])}]"
        lines.append(f"| {row['family']} / {row['prior']} / g{row['grid']} | {row['split']} / {row['horizon']} | {row['baseline']} | {row['label']} | {relative} |")
    lines += ["", "## Calibration and optimizer diagnostics", "",
              "| Setting | Selected candidate | Multipliers | Search bounds | Selected validation NMSE | Calibration elapsed, s | Boundary coordinates |",
              "| --- | --- | --- | --- | ---: | ---: | --- |"]
    for job in job_reports:
        factors = ", ".join(_format(v, 6) for v in job["selected_multipliers"])
        boundary = job["calibration_diagnostics"].get("selected_boundary_hits") or []
        bounds = job["calibration_settings"].get("multiplicative_bounds", [])
        lines.append(f"| {job['family']} / {job['prior']} / g{job['grid']} | {job['selected_candidate']} | {factors} | {bounds} | {_format(job['validation_nmse'])} | {_format(job['calibration_total_seconds'])} | {boundary} |")
    lines += ["", "The complete JSON includes every candidate, optimizer termination, finite-difference call count, boundary hit, numerical failure, and Jacobian rank. A boundary hit or exhausted evaluation budget is a diagnostic. No scientific threshold is attached to either. Coefficient-error experiments admit an exact shared multiplicative correction; the structural cases retain their missing terms.", "",
              "## Computation", "", f"Current elapsed time: {_format(elapsed)} s. Estimated Pod compute cost: {'unavailable (rate or elapsed time missing)' if estimated_cost is None else '$' + format(estimated_cost, '.2f')}.", "",
              "Historical neural training and inference measurements remain identified as historical in the accuracy CSV. Current solver-calibration time cannot establish a matched-hardware training-cost comparison with those records.", ""]
    if timing_rows:
        lines += ["Fresh matched measurements execute resident inputs on one device. Neural weights are initialized because trained checkpoints were omitted from the release. These are architecture-execution measurements; no fresh neural accuracy is inferred. Full block timings and environments are in `matched_timing.csv` and the timing JSON files.", "",
                  "| Setting | Device | Method | Batch | Median ms / batch-step |",
                  "| --- | --- | --- | ---: | ---: |"]
        for row in timing_rows:
            lines.append(f"| {row.get('family')} / {row.get('prior')} / g{row.get('grid')} | {row.get('device')} | {row.get('kind')} | {row.get('batch_size')} | {_format(row.get('median_ms_per_batch_step'))} |")
    else:
        lines += ["No fresh matched timing records were found. No physics-versus-neural latency ratio is inferred."]
    lines += ["", "## Provenance and limits", "", report["selection_scope"], "", report["scoring"], "",
              "Calibration-file hashes must match the freeze; the freeze must match the protocol; evaluation arrays and their linked calibration/freeze must match their recorded hashes. Frozen choices are reconstructed from the validation scores and the declared tie order. Full/pilot evaluations require 64 trajectories per distribution, and smoke requires eight.", "",
              "Incompatible archived outcomes are withheld from neural comparisons. Compatibility requires the original solver source, full counts, ordered trajectory IDs and matching normalization, together with exact development/test archive data hashes or successful numerical reference replay. Exact data hashes establish identical forecast targets even if CPU physical forecasts differ from archived GPU forecasts through floating-point rounding. These replay deviations remain visible in each compatibility record; tolerances are unchanged.", "",
              "Fully observed, noiseless, one-dimensional simulated fields and known forcing define the evaluated use case.", ""]
    if errors or pending or notices:
        lines += ["## Diagnostics requiring attention", ""]
        lines.extend(f"- Integrity: {item['job_id']}: {item['reason']}" for item in errors)
        lines.extend(f"- Pending: {item['job_id']}: {item['reason']}" for item in pending)
        lines.extend(f"- {notice}" for notice in notices)
        lines.append("")
    (reports / "report.md").write_text("\n".join(lines) + "\n")
    print("\nCALIBRATED PHYSICS RESULTS", flush=True)
    print(f"COMPLETION: {completion} | {verified}/{len(requested)} settings verified", flush=True)
    print(f"Scientific evidence: {scientific_status}", flush=True)
    print("No minimum improvement requirement; no required count of successful systems.", flush=True)
    print("GREEN=lower candidate error | YELLOW=inconclusive | RED=higher candidate error | BLUE=identical scores", flush=True)
    for row in comparisons:
        if row["candidate"] != "calibrated":
            continue
        print(f"{row['family']}/{row['prior']} g{row['grid']} {row['split']} H{row['horizon']} | "
              f"CALIBRATED vs {row['baseline']} | {row['label']} | MSE reduction "
              f"{_format(row['mse_reduction_percent'])}% [95% CI {_format(row['reduction_ci95_low'])}, {_format(row['reduction_ci95_high'])}]", flush=True)
    print(f"TIME: elapsed {_format(elapsed)} s; calibration {_format(execution.get('calibration_wall_seconds'))} s; "
          f"data {_format(execution.get('data_seconds'))} s; evaluation {_format(execution.get('evaluation_seconds'))} s.", flush=True)
    if estimated_cost is not None:
        print(f"Estimated Pod compute cost: ${estimated_cost:.2f}; storage and idle time excluded.", flush=True)
    print("Intervals are exploratory, pointwise, and conditional on the fitted solver; no multiplicity adjustment.", flush=True)
    for notice in notices:
        print(f"NOTICE: {notice}", flush=True)
    for item in errors:
        print(f"INTEGRITY ERROR: {item['job_id']}: {item['reason']}", flush=True)
    print(f"Reports: {reports / 'report.md'} and {reports / 'report.json'}", flush=True)
    return _jsonable(report)
