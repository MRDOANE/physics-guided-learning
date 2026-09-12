"""Prospective augmentation selection with physical-family grouped validation.

This module accepts training diagnostics and validation losses only.  It never
opens an evaluation file.  The caller must freeze its output before evaluating
test trajectories or continuing the confirmation-family candidates.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from collections import defaultdict

import numpy as np


ARMS = ("none", "iid", "smooth", "response_matched")
KINDS = ("transformer", "looped", "fno")
FEATURES = ("gap_relative", "gain_iid", "gain_smooth", "low_power", "high_power", "increment_rms")
DEVELOPMENT_FAMILIES = ("wave", "burgers", "ks", "advection_diffusion", "allen_cahn")
RULES = ("none", "always_smooth", "best_fixed", "random", "hand_rule", "probe_best", "ridge_selector")
LOSS_CAP = 1e6
LOSS_FLOOR = 1e-12
_RECORD_KEYS = {
    "task_id", "family", "kind", "prior", "seed", "arm", "features", "probe_val", "final_val",
    "probe_seconds", "remaining_seconds", "setup_seconds", "train_seconds", "diagnostic_seconds",
}


def _loss(value, label):
    """Failing/nonfinite numerical outcomes count as capped loss, not missing rows."""
    if value is None:
        raise ValueError(f"Missing required {label}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {label}: expected a numerical loss") from exc
    if not math.isfinite(number):
        return LOSS_CAP
    if number < 0:
        raise ValueError(f"Negative {label}")
    return max(LOSS_FLOOR, min(LOSS_CAP, number))


def _seconds(record, key):
    value = record.get(key)
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid {key} for {record['task_id']}/{record['arm']}")
    return value


def _settings(cfg):
    options = cfg.get("selector", {})
    dev = tuple(options.get("development_families", DEVELOPMENT_FAMILIES))
    confirmation = str(options.get("confirmation_family", "gray_scott"))
    if len(dev) < 2 or len(set(dev)) != len(dev) or confirmation in dev:
        raise ValueError("Use at least two distinct development families and one disjoint confirmation family")
    if tuple(options.get("arms", ARMS)) != ARMS:
        raise ValueError(f"The prespecified selector requires arms in this order: {ARMS}")
    alphas = tuple(float(value) for value in options.get("ridge_alphas", (0.1, 1, 10, 100)))
    if not alphas or any(not math.isfinite(value) or value <= 0 for value in alphas):
        raise ValueError("Ridge alphas must be finite and positive")
    if tuple(sorted(set(alphas))) != alphas:
        raise ValueError("Ridge alphas must be unique and sorted")
    gain = float(options.get("min_predicted_gain", 0.05))
    if not math.isfinite(gain) or not 0 < gain < 1:
        raise ValueError("min_predicted_gain must be between zero and one")
    return dev, confirmation, alphas, gain, int(options.get("random_seed", 260911))


def _tasks(records, dev, confirmation):
    grouped = defaultdict(dict)
    for original in records:
        extra = set(original) - _RECORD_KEYS
        if extra:
            raise ValueError(f"Unexpected selector record fields (test fields are forbidden): {sorted(extra)}")
        required = {"task_id", "family", "kind", "prior", "seed", "arm", "features", "probe_val"}
        if not required <= original.keys():
            raise ValueError(f"Incomplete selector record; missing {sorted(required - original.keys())}")
        record = dict(original)
        task_id, arm = str(record["task_id"]), record["arm"]
        if not task_id or arm not in ARMS or record["kind"] not in KINDS:
            raise ValueError("Invalid task_id, arm, or architecture")
        if record["family"] not in set(dev) | {confirmation}:
            raise ValueError(f"Unscheduled physical family: {record['family']}")
        if arm in grouped[task_id]:
            raise ValueError(f"Duplicate candidate: {task_id}/{arm}")
        if set(record["features"]) != set(FEATURES):
            raise ValueError(f"Features must be exactly {FEATURES}; no family, prior, seed or labels")
        try:
            features = {name: float(record["features"][name]) for name in FEATURES}
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid training diagnostics for {task_id}") from exc
        if not all(math.isfinite(value) for value in features.values()):
            raise ValueError(f"Nonfinite training diagnostics for {task_id}; repair diagnostics before selecting")
        record["task_id"] = task_id
        record["features"] = features
        record["probe_val"] = _loss(record["probe_val"], f"probe_val {task_id}/{arm}")
        if record["family"] in dev:
            record["final_val"] = _loss(record.get("final_val"), f"final_val {task_id}/{arm}")
        else:
            # This is intentionally discarded even if a caller accidentally has
            # confirmation outcomes in memory.  They cannot change the choices.
            record.pop("final_val", None)
        for key in ("probe_seconds", "remaining_seconds", "setup_seconds", "train_seconds", "diagnostic_seconds"):
            record[key] = _seconds(record, key)
        grouped[task_id][arm] = record
    if not grouped:
        raise ValueError("No candidate records")
    result = []
    family_names = set()
    for task_id in sorted(grouped):
        candidates = grouped[task_id]
        if set(candidates) != set(ARMS):
            raise ValueError(f"Incomplete candidate bank for {task_id}: expected all four arms")
        base = candidates["none"]
        for arm in ARMS:
            current = candidates[arm]
            for key in ("family", "kind", "prior", "seed", "features", "diagnostic_seconds"):
                if current[key] != base[key]:
                    raise ValueError(f"Candidate metadata/diagnostics disagree for {task_id}: {key}")
        family_names.add(base["family"])
        result.append({"task_id": task_id, "family": base["family"], "kind": base["kind"],
                       "prior": base["prior"], "seed": base["seed"], "features": base["features"],
                       "candidates": candidates})
    expected = set(dev) | {confirmation}
    if family_names != expected:
        raise ValueError(f"Incomplete family inventory: missing {sorted(expected - family_names)}")
    # If supplied, the experiment inventory is enforced by the caller/protocol.
    # Here we additionally prevent an accidental duplicated scientific task.
    signatures = [(t["family"], t["kind"], t["prior"], t["seed"]) for t in result]
    if len(signatures) != len(set(signatures)):
        raise ValueError("Different task IDs refer to the same family/architecture/prior/seed")
    return result


def feature_names():
    """Stable design-matrix order; all standardization is learned within a fold."""
    return ([f"diagnostic:{name}" for name in FEATURES]
            + [f"architecture:{kind}" for kind in KINDS]
            + [f"arm:{arm}" for arm in ARMS[1:]]
            + ["log_probe_vs_none"]
            + [f"{arm}*{name}" for arm in ARMS[1:] for name in FEATURES]
            + [f"{arm}*log_probe_vs_none" for arm in ARMS[1:]])


def _x(task, arm):
    common = [task["features"][name] for name in FEATURES]
    indicators = [float(arm == item) for item in ARMS[1:]]
    probe_relative = math.log(task["candidates"][arm]["probe_val"] / task["candidates"]["none"]["probe_val"])
    return np.asarray(common + [float(task["kind"] == kind) for kind in KINDS] + indicators
                      + [probe_relative]
                      + [flag * value for flag in indicators for value in common]
                      + [flag * probe_relative for flag in indicators], dtype=np.float64)


def _rows(tasks):
    x, y, families = [], [], []
    for task in tasks:
        reference = task["candidates"]["none"]["final_val"]
        for arm in ARMS[1:]:
            x.append(_x(task, arm))
            y.append(math.log(task["candidates"][arm]["final_val"] / reference))
            families.append(task["family"])
    if not x:
        raise ValueError("Cannot fit a selector without development candidates")
    return np.stack(x), np.asarray(y, dtype=np.float64), np.asarray(families)


def _weights(families):
    names, counts = np.unique(families, return_counts=True)
    lookup = dict(zip(names, counts))
    # Equal total weight per family, with mean sample weight one so that the
    # prespecified ridge penalty has a conventional sum-of-squares scale.
    return np.asarray([len(families) / (len(names) * lookup[name]) for name in families])


def _fit(tasks, alpha):
    x, y, families = _rows(tasks)
    weights = _weights(families)
    mean = np.average(x, axis=0, weights=weights)
    scale = np.sqrt(np.average((x - mean) ** 2, axis=0, weights=weights))
    scale[scale < 1e-12] = 1.0
    z = (x - mean) / scale
    intercept = float(np.average(y, weights=weights))
    lhs = z.T @ (weights[:, None] * z) + float(alpha) * np.eye(x.shape[1])
    rhs = z.T @ (weights * (y - intercept))
    coefficients = np.linalg.solve(lhs, rhs)
    if not np.isfinite(coefficients).all():
        raise ValueError("Nonfinite ridge coefficients; cannot freeze choices")
    return {"alpha": float(alpha), "mean": mean.tolist(), "scale": scale.tolist(),
            "coefficients": coefficients.tolist(), "intercept": intercept,
            "training_families": sorted(set(families.tolist())), "training_tasks": len(tasks),
            "training_rows": len(y)}


def _predict(model, task, arm):
    if arm == "none":
        return 0.0
    value = model["intercept"] + (((_x(task, arm) - np.asarray(model["mean"]))
                                  / np.asarray(model["scale"])) @ np.asarray(model["coefficients"]))
    if not math.isfinite(float(value)):
        raise ValueError("Nonfinite selector prediction")
    return float(value)


def _nested_alpha(tasks, alphas):
    families = sorted({task["family"] for task in tasks})
    if len(families) < 2:
        fallback = min(alphas, key=lambda value: (abs(math.log(value / 10.0)), value))
        return fallback, {"mode": "smoke_only_fixed_alpha_insufficient_inner_families", "folds": [], "scores": {}}
    scores, fold_metadata = {}, []
    for held_out in families:
        fold_metadata.append({"held_out_family": held_out,
                              "training_families": [name for name in families if name != held_out]})
    for alpha in alphas:
        errors = []
        for held_out in families:
            training = [task for task in tasks if task["family"] != held_out]
            validation = [task for task in tasks if task["family"] == held_out]
            model = _fit(training, alpha)
            residuals = []
            for task in validation:
                reference = task["candidates"]["none"]["final_val"]
                for arm in ARMS[1:]:
                    target = math.log(task["candidates"][arm]["final_val"] / reference)
                    residuals.append((_predict(model, task, arm) - target) ** 2)
            errors.append(float(np.mean(residuals)))
        scores[str(float(alpha))] = float(np.mean(errors))
    best = min(alphas, key=lambda value: (scores[str(float(value))], value))
    return best, {"mode": "leave_one_training_family_out", "folds": fold_metadata, "scores": scores}


def _best_fixed(training):
    family_losses = defaultdict(lambda: defaultdict(list))
    for task in training:
        reference = task["candidates"]["none"]["final_val"]
        for arm in ARMS:
            family_losses[task["family"]][arm].append(math.log(task["candidates"][arm]["final_val"] / reference))
    scores = {arm: float(np.mean([np.mean(values[arm]) for values in family_losses.values()])) for arm in ARMS}
    return min(ARMS, key=lambda arm: scores[arm]), scores


def _random_arm(task_id, seed):
    payload = f"physics-family-selector/random/v1/{seed}/{task_id}".encode()
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return ARMS[integer % len(ARMS)]


def _sum_known(values):
    return None if any(value is None for value in values) else float(sum(values))


def _costs(task, choices):
    result = {}
    for rule, chosen in choices.items():
        probes = list(ARMS) if rule in ("ridge_selector", "probe_best") else []
        acquired = probes or [chosen]
        setup = _sum_known([task["candidates"][arm]["setup_seconds"] for arm in acquired])
        needs_diagnostics = rule in ("ridge_selector", "hand_rule", "probe_best") or chosen != "none"
        diagnostic = task["candidates"]["none"]["diagnostic_seconds"] if needs_diagnostics else 0.0
        if probes:
            probe_seconds = _sum_known([task["candidates"][arm]["probe_seconds"] for arm in probes])
            remaining = task["candidates"][chosen]["remaining_seconds"]
            training = _sum_known([probe_seconds, remaining])
        else:
            probe_seconds = 0.0
            remaining = None
            # At the freeze barrier a confirmation candidate may have only
            # its probe prefix in train_seconds. Do not call that a full fit.
            training = (task["candidates"][chosen]["train_seconds"]
                        if task["candidates"][chosen]["remaining_seconds"] is not None else None)
        result[rule] = {"probe_arms": probes, "selected_arm": chosen,
                        "reuse_selected_probe": bool(probes), "setup_seconds": setup,
                        "diagnostic_seconds": diagnostic,
                        "probe_seconds": probe_seconds, "selected_remaining_seconds": remaining,
                        "target_training_seconds": training,
                        "total_target_seconds": _sum_known([setup, training, diagnostic])}
    return result


def freeze_choices(records, cfg):
    """Return a JSON-safe manifest, using no target-family final validation labels.

    The full development candidate bank is required. Each decision uses a fit
    excluding that physical family; the locked confirmation decisions use all
    development families. `final_val` for confirmation is discarded immediately.
    All test fields and unknown record fields are rejected at the API boundary.
    """
    dev, confirmation, alphas, gain, random_seed = _settings(cfg)
    tasks = _tasks(list(records), dev, confirmation)
    if all(key in cfg for key in ("kinds", "priors", "seeds")):
        expected = {(family, kind, prior, seed) for family in (*dev, confirmation)
                    for kind in cfg["kinds"] for prior in cfg["priors"] for seed in cfg["seeds"]}
        actual = {(task["family"], task["kind"], task["prior"], task["seed"]) for task in tasks}
        if actual != expected:
            raise ValueError(f"Selector inventory differs from configured schedule: {len(expected - actual)} missing, {len(actual - expected)} unexpected tasks")
    models, decisions = {}, []
    for target_family in (*dev, confirmation):
        training = [task for task in tasks if task["family"] in dev and task["family"] != target_family]
        started = time.perf_counter()
        alpha, cross_validation = _nested_alpha(training, alphas)
        model = _fit(training, alpha)
        best_fixed, fixed_scores = _best_fixed(training)
        fixed_by_kind, fixed_scores_by_kind = {}, {}
        for kind in KINDS:
            architecture_tasks = [task for task in training if task["kind"] == kind]
            if architecture_tasks:
                fixed_by_kind[kind], fixed_scores_by_kind[kind] = _best_fixed(architecture_tasks)
        model_id = f"heldout_{target_family}" if target_family in dev else "locked_confirmation"
        models[model_id] = {**model, "feature_names": feature_names(), "inner_cv": cross_validation,
                            "best_fixed_arm": best_fixed, "best_fixed_scores": fixed_scores,
                            "best_fixed_by_kind": fixed_by_kind,
                            "best_fixed_scores_by_kind": fixed_scores_by_kind,
                            "fit_seconds": time.perf_counter() - started}
        for task in tasks:
            if task["family"] != target_family:
                continue
            predictions = {arm: _predict(model, task, arm) for arm in ARMS}
            selected = min(ARMS, key=lambda arm: predictions[arm])
            if predictions[selected] > math.log1p(-gain):
                selected = "none"
            diagnostics = task["features"]
            if task["kind"] not in fixed_by_kind:
                raise ValueError(f"No outer-training candidates for architecture {task['kind']}")
            choices = {"none": "none", "always_smooth": "smooth", "best_fixed": fixed_by_kind[task["kind"]],
                       "random": _random_arm(task["task_id"], random_seed),
                       "hand_rule": "smooth" if diagnostics["gap_relative"] < 0.25
                       and diagnostics["gain_smooth"] <= diagnostics["gain_iid"] else "none",
                       "probe_best": min(ARMS, key=lambda arm: task["candidates"][arm]["probe_val"]),
                       "ridge_selector": selected}
            decisions.append({"task_id": task["task_id"], "family": target_family,
                              "kind": task["kind"], "prior": task["prior"], "seed": task["seed"],
                              "role": "locked_confirmation" if target_family == confirmation else "development_family_holdout",
                              "model_id": model_id, "choices": choices,
                              "diagnostic_seconds": task["candidates"]["none"]["diagnostic_seconds"],
                              "predicted_log_relative_loss": predictions,
                              "costs": _costs(task, choices)})
    return {"schema": 1, "method": "family_grouped_ridge_with_abstention_v1", "frozen": True,
            "development_families": list(dev), "confirmation_family": confirmation,
            "arms": list(ARMS), "rules": list(RULES), "min_predicted_gain": gain,
            "random_seed": random_seed, "loss_cap": LOSS_CAP, "loss_floor": LOSS_FLOOR,
            "models": models,
            "fit_metadata": [{"model_id": model_id,
                              "held_out_family": confirmation if model_id == "locked_confirmation" else model_id.removeprefix("heldout_"),
                              "training_families": model["training_families"],
                              "training_tasks": model["training_tasks"], "alpha": model["alpha"],
                              "inner_cv": model["inner_cv"], "fit_seconds": model["fit_seconds"]}
                             for model_id, model in models.items()],
            "selector_fit_seconds": float(sum(model["fit_seconds"] for model in models.values())),
            "decisions": sorted(decisions, key=lambda item: item["task_id"]),
            "information_policy": {
                "features": "common training diagnostics, architecture, candidate identity, short-probe validation loss relative to none",
                "labels": "development-only full-budget validation loss relative to none",
                "excluded": ["test data", "test losses", "family identity as feature", "prior identity as feature",
                             "seed as feature", "target-family final validation labels"],
                "costs": "Per-target costs; selector development bank and offline full-candidate reference costs must be reported separately.",
            }}


def manifest_digest(manifest):
    """Hash a frozen manifest for its phase barrier (including its recorded costs)."""
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
