"""Fit a small physical-parameter correction using train/validation data only.

The estimator has three shared parameters, one per supplied physical coefficient:
    calibrated_parameters = supplied_parameters * exp(log_multipliers).
The correction is shared by all trajectories in a setting.  It does not use
true coefficients, restore omitted equation terms, or adapt on test trajectories.

``step_fn`` receives NumPy state/action/parameter arrays and the keyword arguments
``mismatch`` and ``fidelity``.  The caller owns data provenance, the test-access
barrier, freezing, timing comparisons, and final statistical reporting.
"""
from __future__ import annotations

import math
import time
from typing import Callable

import numpy as np
from scipy.optimize import least_squares


CONTEXT = 4
LOSS_CAP = 1_000_000.0
DIVERGENCE_THRESHOLD = 1_000_000.0
MULTIPLIER_BOUNDS = (0.001, 1000.0)
STARTS = ((1.0, 1.0, 1.0), (0.8, 1.2, 1.0), (1.2, 0.8, 1.0))


def _dtype(value):
    result = np.dtype(np.float64 if value is None else value)
    if result not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("numerical_dtype must be float32 or float64")
    return result


def _split_arrays(split, dtype=np.float64):
    """Read only the three permitted fields; check complete trajectories."""
    states = np.asarray(split["states"], dtype=dtype)
    actions = np.asarray(split["actions"], dtype=dtype)
    params = np.asarray(split["supplied_params"], dtype=dtype)
    if states.ndim != 4 or actions.ndim != 3 or params.ndim != 2:
        raise ValueError("Expected states [N,T+1,C,G], actions [N,T,1], supplied_params [N,3]")
    n = states.shape[0]
    if n < 1 or actions.shape != (n, states.shape[1] - 1, 1) or params.shape != (n, 3):
        raise ValueError("Incompatible trajectory shapes or empty split")
    if not all(np.isfinite(a).all() for a in (states, actions, params)):
        raise ValueError("Input trajectories and supplied parameters must be finite")
    if np.any(params <= 0):
        raise ValueError("The declared multiplicative estimator requires positive supplied coefficients")
    return states, actions, params


def _channel_values(value, channels, dtype, name, *, positive=False):
    array = np.asarray(value, dtype=dtype).reshape(-1)
    if array.shape != (channels,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain one finite value per state channel")
    if positive and np.any(array <= 0):
        raise ValueError(f"{name} must be strictly positive")
    return array[None, :, None]


def _rollout_impl(step_fn, family, split, state_scale, prior, multipliers,
                  horizon, numerical_dtype, state_mean):
    dtype = _dtype(numerical_dtype)
    states, actions, supplied = _split_arrays(split, dtype)
    horizon = int(horizon)
    if horizon < 1 or CONTEXT - 1 + horizon > actions.shape[1]:
        raise ValueError("Requested rollout exceeds available actions or has no steps")
    factors = np.asarray(multipliers, dtype=dtype)
    if factors.shape != (3,) or not np.isfinite(factors).all() or np.any(factors <= 0):
        raise ValueError("multipliers must be three positive finite values")
    scale = _channel_values(state_scale, states.shape[2], dtype, "state_scale", positive=True)
    mean = _channel_values(np.zeros(states.shape[2]) if state_mean is None else state_mean,
                           states.shape[2], dtype, "state_mean")
    params = np.asarray(supplied * factors[None], dtype=dtype)
    prediction = states[:, CONTEXT - 1].copy()
    failed = np.zeros(len(states), dtype=bool)
    curve = np.empty((len(states), horizon), dtype=np.float64)
    failure_reasons = []
    for h in range(horizon):
        try:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                new = np.asarray(step_fn(family, prediction, actions[:, CONTEXT - 1 + h], params,
                                         mismatch=prior, fidelity="coarse"), dtype=dtype)
            if new.shape != prediction.shape:
                raise ValueError("step_fn returned an unexpected state shape")
        except (FloatingPointError, OverflowError, RuntimeError, ValueError) as exc:
            failed[:] = True
            new = np.broadcast_to(mean, prediction.shape).copy()
            failure_reasons.append(f"step {h + 1}: {type(exc).__name__}: {exc}")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            normalized = (new - mean) / scale
            bad = ~np.isfinite(normalized).all(axis=(1, 2))
            bad |= np.max(np.abs(normalized), axis=(1, 2)) > DIVERGENCE_THRESHOLD
            failed |= bad
            errors = np.mean(((new - states[:, CONTEXT + h]) / scale) ** 2, axis=(1, 2))
        failed |= ~np.isfinite(errors)
        errors[failed] = np.inf
        curve[:, h] = errors
        # Every subsequent error of a failed trajectory stays infinite. This
        # replacement only keeps independent batched arithmetic usable.
        prediction = np.where(failed[:, None, None], mean, new).astype(dtype, copy=False)
    return curve, failed, failure_reasons


def rollout_error(step_fn: Callable, family: str, split: dict, state_scale,
                  prior: str, multipliers, horizon: int = 96,
                  numerical_dtype=None, state_mean=None):
    """Return ``(per_step_nmse[N,H], failed_trajectory_count)``.

    Rollouts begin at observed state index 3 and use the action at index 3 for
    their first forecast, matching the archived four-state context. The output
    contains raw errors, including infinity after divergence; the caller applies
    the source stage's scoring rule. ``numerical_dtype='float32'`` matches the
    archived arithmetic precision. No true coefficients are read.
    """
    curve, failed, _ = _rollout_impl(step_fn, family, split, state_scale, prior,
                                      multipliers, horizon, numerical_dtype, state_mean)
    return curve, int(failed.sum())


def _windows(train, horizon, per_trajectory, state_scale, state_mean):
    states, actions, params = _split_arrays(train, np.float64)
    first, last = CONTEXT - 1, actions.shape[1] - horizon
    if last < first:
        raise ValueError("Training trajectory is too short for the requested objective")
    times = np.unique(np.linspace(first, last, per_trajectory, dtype=np.int64))
    ids = np.repeat(np.arange(len(states)), len(times))
    starts = np.tile(times, len(states))
    scale = _channel_values(state_scale, states.shape[2], np.float64, "state_scale", positive=True)
    mean = _channel_values(np.zeros(states.shape[2]) if state_mean is None else state_mean,
                           states.shape[2], np.float64, "state_mean")
    return {
        "initial": states[ids, starts].copy(),
        "actions": np.stack([actions[ids, starts + h] for h in range(horizon)], axis=1),
        "targets": np.stack([states[ids, starts + h + 1] for h in range(horizon)], axis=1),
        "params": params[ids].copy(), "scale": scale, "mean": mean,
        "start_indices": times.tolist(), "count": len(ids),
    }


def _residual_function(step_fn, family, prior, windows, counters):
    shape = windows["targets"].shape
    size = int(np.prod(shape))
    root_size = math.sqrt(size)
    penalty = np.full(size, math.sqrt(LOSS_CAP) / root_size, dtype=np.float64)

    def residual(log_multipliers):
        counters["residual_calls"] += 1
        prediction = windows["initial"].copy()
        params = windows["params"] * np.exp(log_multipliers)[None]
        residuals = np.empty(shape, dtype=np.float64)
        try:
            for h in range(shape[1]):
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    prediction = np.asarray(step_fn(family, prediction, windows["actions"][:, h], params,
                                                     mismatch=prior, fidelity="coarse"), dtype=np.float64)
                    if prediction.shape != windows["initial"].shape:
                        raise ValueError("step_fn returned an unexpected state shape")
                    normalized = (prediction - windows["mean"]) / windows["scale"]
                    if not np.isfinite(normalized).all() or np.max(np.abs(normalized)) > DIVERGENCE_THRESHOLD:
                        raise FloatingPointError("nonfinite or divergent training rollout")
                    residuals[:, h] = (prediction - windows["targets"][:, h]) / windows["scale"]
            if not np.isfinite(residuals).all():
                raise FloatingPointError("nonfinite normalized training residual")
        except (FloatingPointError, OverflowError, RuntimeError, ValueError) as exc:
            counters["invalid_residual_calls"] += 1
            counters["last_residual_valid"] = False
            reason = f"{type(exc).__name__}: {exc}"
            if reason not in counters["failure_reasons"] and len(counters["failure_reasons"]) < 8:
                counters["failure_reasons"].append(reason)
            return penalty.copy()
        counters["last_residual_valid"] = True
        return residuals.reshape(-1) / root_size

    return residual


def fit_setting(step_fn: Callable, family: str, train: dict, val: dict,
                state_scale, prior: str, profile: str = "full", *,
                validation_step_fn: Callable | None = None, state_mean=None) -> dict:
    """Fit and select a calibrated solver; return JSON-compatible diagnostics.

    Full/pilot settings use all 128 training trajectories, two evenly spaced
    windows from each trajectory, training horizons 1 and 4, three generic starts,
    bounded three-point finite-difference least squares, and at most 60 SciPy
    function evaluations per fit. Numerical Jacobian calls are counted separately
    by ``residual_calls``. Smoke uses the first eight training trajectories, the
    first four validation trajectories, horizon 1, identity start and a budget of
    15 evaluations. Smoke results are never scientific evidence.

    Validation uses the original 16-step recursive forecast with float32 inputs.
    An unchanged identity candidate is always included. Failed validation
    trajectories remain in scoring at 1e6 each, matching the archived rule.
    Optimizer budget exhaustion is retained as a diagnostic: a finite candidate
    can still be selected.
    """
    started = time.perf_counter()
    if profile not in ("full", "pilot", "smoke"):
        raise ValueError("profile must be full, pilot, or smoke")
    if prior not in ("correct", "coefficient", "structural"):
        raise ValueError("Unknown physical prior")
    train_arrays = _split_arrays(train)
    val_arrays = _split_arrays(val)
    if profile != "smoke" and (len(train_arrays[0]) != 128 or len(val_arrays[0]) != 24):
        raise ValueError("Full/pilot calibration requires the archived 128 train and 24 validation trajectories")
    smoke = profile == "smoke"
    train_count = min(8, len(train_arrays[0])) if smoke else len(train_arrays[0])
    val_count = min(4, len(val_arrays[0])) if smoke else len(val_arrays[0])
    permitted = ("states", "actions", "supplied_params")
    train_used = {key: np.asarray(train[key])[:train_count] for key in permitted}
    val_used = {key: np.asarray(val[key])[:val_count] for key in permitted}
    starts = STARTS[:1] if smoke else STARTS
    horizons = (1,) if smoke else (1, 4)
    max_nfev = 15 if smoke else 60
    validation_step = validation_step_fn or step_fn
    candidates = []
    fit_seconds = validation_seconds = 0.0

    def validate(factors):
        begin = time.perf_counter()
        curve, failed, reasons = _rollout_impl(validation_step, family, val_used, state_scale,
                                               prior, factors, 16, "float32", state_mean)
        # The archive caps each complete trajectory's mean before averaging.
        # One divergent trajectory therefore contributes LOSS_CAP / N, without
        # replacing the entire setting's score or excluding the candidate.
        with np.errstate(over="ignore", invalid="ignore"):
            means = np.nan_to_num(curve.mean(axis=1), nan=LOSS_CAP, posinf=LOSS_CAP, neginf=LOSS_CAP)
        score = float(np.minimum(means, LOSS_CAP).mean())
        valid = bool(np.isfinite(score))
        return score, valid, int(failed.sum()), reasons, time.perf_counter() - begin

    score, valid, failed, reasons, seconds = validate([1.0, 1.0, 1.0])
    validation_seconds += seconds
    candidates.append({
        "candidate_id": "identity", "kind": "uncalibrated_fallback", "multipliers": [1.0, 1.0, 1.0],
        "log_multipliers": [0.0, 0.0, 0.0], "validation_nmse": score, "validation_valid": valid,
        "validation_failed_trajectories": failed, "validation_failure_reasons": reasons,
        "validation_seconds": seconds, "fit_seconds": 0.0, "residual_calls": 0,
        "invalid_residual_calls": 0, "boundary_hits": [], "optimizer_success": None,
    })
    setup_seconds = time.perf_counter() - started - validation_seconds
    lower, upper = np.log(MULTIPLIER_BOUNDS)
    for horizon in horizons:
        window_started = time.perf_counter()
        windows = _windows(train_used, horizon, 2, state_scale, state_mean)
        setup_seconds += time.perf_counter() - window_started
        for start_index, initial in enumerate(starts):
            counters = {"residual_calls": 0, "invalid_residual_calls": 0,
                        "last_residual_valid": None, "failure_reasons": []}
            objective = _residual_function(step_fn, family, prior, windows, counters)
            begin = time.perf_counter()
            row = {
                "candidate_id": f"h{horizon}_start{start_index}", "kind": "calibrated",
                "training_horizon": horizon, "training_window_start_indices": windows["start_indices"],
                "training_window_count": windows["count"], "initial_multipliers": list(initial),
                "multipliers": None, "validation_nmse": LOSS_CAP, "validation_valid": False,
                "validation_failed_trajectories": None, "validation_failure_reasons": [],
                "validation_seconds": 0.0, "optimizer_success": False, "boundary_hits": [],
                "jacobian_rank": None, "jacobian_condition_number": None,
            }
            try:
                result = least_squares(objective, np.log(initial), bounds=(lower, upper),
                                       jac="3-point", method="trf", max_nfev=max_nfev,
                                       ftol=1e-8, xtol=1e-8, gtol=1e-8)
                factors = np.exp(result.x)
                final_residual = objective(result.x)
                final_valid = bool(counters["last_residual_valid"])
                singular = np.linalg.svd(result.jac, compute_uv=False)
                tolerance = max(result.jac.shape) * np.finfo(float).eps * (singular[0] if len(singular) else 0.0)
                rank = int(np.count_nonzero(singular > tolerance))
                condition = float(singular[0] / singular[-1]) if len(singular) and singular[-1] > 0 else None
                row.update({
                    "multipliers": factors.tolist(), "log_multipliers": result.x.tolist(),
                    "optimizer_success": bool(result.success), "optimizer_status": int(result.status),
                    "optimizer_message": str(result.message), "optimizer_nfev": int(result.nfev),
                    "optimizer_njev": None if result.njev is None else int(result.njev),
                    "optimizer_optimality": float(result.optimality) if np.isfinite(result.optimality) else None,
                    "training_nmse": float(np.dot(final_residual, final_residual)),
                    "final_training_valid": final_valid, "jacobian_rank": rank,
                    "jacobian_singular_values": singular.tolist(),
                    "jacobian_condition_number": condition if condition is None or np.isfinite(condition) else None,
                    "boundary_hits": [i for i, x in enumerate(result.x)
                                      if abs(x - lower) <= 1e-5 or abs(x - upper) <= 1e-5],
                })
            except (FloatingPointError, OverflowError, RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
                row["optimizer_exception"] = f"{type(exc).__name__}: {exc}"
            row["fit_seconds"] = time.perf_counter() - begin
            fit_seconds += row["fit_seconds"]
            row.update({key: counters[key] for key in ("residual_calls", "invalid_residual_calls", "failure_reasons")})
            if row["multipliers"] is not None and row.get("final_training_valid", False):
                score, valid, failed, reasons, seconds = validate(row["multipliers"])
                validation_seconds += seconds
                row.update(validation_nmse=score, validation_valid=valid,
                           validation_failed_trajectories=failed, validation_failure_reasons=reasons,
                           validation_seconds=seconds)
            candidates.append(row)

    eligible = [row for row in candidates if row["validation_valid"] and np.isfinite(row["validation_nmse"])]
    # Stable ordering makes exact ties select identity first, then earlier
    # predeclared objective/start candidates. Test information is unavailable.
    selected = min(eligible, key=lambda row: row["validation_nmse"]) if eligible else candidates[0]
    all_invalid = not bool(eligible)
    return {
        "schema_version": 1, "family": family, "prior": prior, "profile": profile,
        "status": "NO_VALID_CANDIDATE" if all_invalid else "FITTED_AND_VALIDATED",
        "fit_valid": not all_invalid, "scientific_profile": profile != "smoke",
        "estimator": "three positive multipliers shared across training trajectories; coefficients supplied to the solver are multiplied once",
        "structural_omissions_preserved": prior == "structural",
        "neural_training_performed": False, "true_parameters_used_for_fitting": False,
        "test_data_used_for_fitting_or_selection": False,
        "train_trajectories": train_count, "validation_trajectories": val_count,
        "state_scale": np.asarray(state_scale, dtype=float).reshape(-1).tolist(),
        "state_mean": None if state_mean is None else np.asarray(state_mean, dtype=float).reshape(-1).tolist(),
        "settings": {"multiplicative_bounds": list(MULTIPLIER_BOUNDS), "starts": [list(s) for s in starts],
                     "training_horizons": list(horizons), "windows_per_training_trajectory": 2,
                     "validation_horizon": 16, "context": CONTEXT, "fidelity": "coarse",
                     "fit_dtype": "float64", "validation_dtype": "float32",
                     "optimizer": "scipy.optimize.least_squares", "jacobian": "3-point",
                     "max_nfev_per_fit": max_nfev, "ftol": 1e-8, "xtol": 1e-8, "gtol": 1e-8,
                     "finite_failure_penalty_nmse": LOSS_CAP, "divergence_threshold": DIVERGENCE_THRESHOLD},
        "selected_candidate": selected["candidate_id"],
        "selected_multipliers": selected["multipliers"], "multipliers": selected["multipliers"],
        "validation_nmse": selected["validation_nmse"], "selected_validation_nmse": selected["validation_nmse"],
        "identity_validation_nmse": candidates[0]["validation_nmse"],
        "selected_boundary_hits": selected["boundary_hits"],
        "candidate_records": candidates, "candidate_count": len(candidates),
        "valid_candidate_count": len(eligible),
        "residual_calls": sum(row["residual_calls"] for row in candidates),
        "invalid_residual_calls": sum(row["invalid_residual_calls"] for row in candidates),
        "fit_seconds": fit_seconds, "validation_seconds": validation_seconds,
        "setup_seconds": setup_seconds, "total_seconds": time.perf_counter() - started,
        "interpretation": "Validation selects among declared training objectives and starts, including the unchanged solver. Boundary hits and optimizer termination are diagnostics, without a minimum required improvement.",
    }
