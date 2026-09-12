"""Auditable mechanism and prospective selector reporting.

All decisions are supplied by the frozen selection manifest.  This module never
chooses a policy from test results.  Intervals describe this finite experiment;
five development families do not establish a universal selection law.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


ARMS = ("none", "iid", "smooth", "response_matched")
POLICIES = ("ridge_selector", "none", "always_smooth", "best_fixed", "random", "hand_rule", "probe_best", "full_validation")
EPS = 1e-15


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list, tuple)) else v for k, v in row.items()})


def _families(cfg):
    return [v["name"] if isinstance(v, dict) else v for v in cfg["families"]]


def _key(record, include_arm=True):
    key = (str(record["family"]), str(record["kind"]), str(record["prior"]), int(record["seed"]))
    return key + (str(record["arm"]),) if include_arm else key


def _score(curves, horizon, cap):
    """Cap after each trajectory's time mean; never delete a failed trajectory."""
    curves = np.asarray(curves, dtype=np.float64)
    if curves.ndim != 2 or curves.shape[0] == 0 or curves.shape[1] < horizon:
        raise ValueError(f"Expected nonempty trajectory x horizon array with at least {horizon} steps, got {curves.shape}")
    curves = curves[:, :horizon]
    if np.any(curves[np.isfinite(curves)] < 0):
        raise ValueError("Squared error cannot be negative")
    invalid = np.any(~np.isfinite(curves), axis=1)
    with np.errstate(over="ignore", invalid="ignore"):
        raw = np.mean(curves, axis=1)
    invalid |= ~np.isfinite(raw)
    raw[invalid] = np.inf
    capped = np.minimum(raw, cap)
    return capped, invalid, raw > cap


def _load(output, cfg, records):
    errors, bank = [], {}
    horizon = int(cfg.get("primary_horizon", 64))
    cap = float(cfg.get("loss_cap", 1e6))
    families = _families(cfg)
    expected = {(f, k, p, int(s), a) for f in families for k in cfg["kinds"] for p in cfg["priors"] for s in cfg["seeds"] for a in ARMS}
    family_ids = {}
    for record in records:
        try:
            key = _key(record)
            if key in bank:
                raise ValueError(f"Duplicate record {key}")
            if key not in expected:
                raise ValueError(f"Unexpected record {key}")
            for timing in ("setup_seconds", "probe_setup_seconds", "continuation_setup_seconds", "probe_seconds", "remaining_seconds", "train_seconds", "evaluation_seconds", "bank_setup_seconds", "diagnostic_seconds", "inference_ms_per_step"):
                value = float(record.get(timing, 0))
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"Invalid nonnegative timing {timing}")
            path = (output / record["evaluation_path"]).resolve()
            if not path.is_relative_to(output.resolve()):
                raise ValueError("Evaluation path leaves the run directory")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != record.get("evaluation_sha256"):
                raise ValueError(f"Evaluation SHA256 mismatch: {path.name}")
            arrays = {}
            with np.load(path, allow_pickle=False) as bundle:
                for split in ("test", "ood"):
                    curve = np.asarray(bundle[f"{split}_mse"], dtype=np.float64)
                    ids = np.asarray(bundle[f"{split}_ids"]).astype(str)
                    if ids.ndim != 1 or curve.shape[0] != len(ids) or len(set(ids)) != len(ids):
                        raise ValueError(f"Invalid or repeated {split} trajectory IDs")
                    order = np.argsort(ids)
                    ids = ids[order]
                    id_key = (key[0], split)
                    if id_key in family_ids and not np.array_equal(family_ids[id_key], ids):
                        raise ValueError(f"Unpaired {split} trajectory IDs within family {key[0]}")
                    family_ids[id_key] = ids
                    curve = curve[order]
                    score, diverged, clipped = _score(curve, horizon, cap)
                    arrays[split] = {"curve": curve, "score": score, "diverged": diverged, "capped": clipped, "ids": ids}
                    final_name = f"{split}_exact_mse"
                    if final_name in bundle:
                        # "exact" in the engine means the exact final training
                        # step, not an exact physical simulator.
                        final_curve = np.asarray(bundle[final_name], dtype=np.float64)
                        if final_curve.shape[0] != len(order):
                            raise ValueError(f"Mismatched trajectory count for {final_name}")
                        final_curve = final_curve[order]
                        final_score, final_div, final_cap = _score(final_curve, horizon, cap)
                        arrays[f"{split}_final_step"] = {"curve": final_curve, "score": final_score,
                                                        "diverged": final_div, "capped": final_cap, "ids": ids}
                    for baseline in ("persistence", "mechanistic"):
                        name = f"{split}_{baseline}_mse"
                        if name in bundle:
                            base_curve = np.asarray(bundle[name], dtype=np.float64)
                            if base_curve.shape[0] != len(order):
                                raise ValueError(f"Mismatched trajectory count for {name}")
                            arrays[f"{split}_{baseline}"] = _score(base_curve[order], horizon, cap)
            bank[key] = {"record": record, "arrays": arrays}
        except (KeyError, ValueError, OSError, TypeError, IndexError) as exc:
            errors.append(f"{record.get('task_id', 'unknown task')}: {exc}")
    missing = expected - set(bank)
    if missing:
        examples = ", ".join(str(k) for k in sorted(missing)[:5])
        errors.append(f"Missing or invalid {len(missing)} of {len(expected)} candidate records; examples: {examples}")
    return bank, errors, len(expected)


def _interval(samples, multiplicity):
    tail = .05 / (2 * multiplicity)
    return np.quantile(samples, [tail, 1 - tail]).tolist()


def _within_family_bootstrap(a, b, rng, replicates):
    """Arrays [architecture/condition, seed, shared trajectory].

    Crossed paired bootstrap: one seed draw and one trajectory draw are shared
    across all architectures, prior conditions, and both members of a contrast.
    """
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.ndim != 3:
        raise ValueError("Paired bootstrap requires equal condition x seed x trajectory arrays")
    _, seeds, trajectories = a.shape
    point = float(np.log((a.mean(axis=2) + EPS) / (b.mean(axis=2) + EPS)).mean())
    samples = np.empty(replicates, dtype=np.float64)
    probabilities_t = np.full(trajectories, 1 / trajectories)
    probabilities_s = np.full(seeds, 1 / seeds)
    for start in range(0, replicates, 1000):
        count = min(1000, replicates - start)
        tw = rng.multinomial(trajectories, probabilities_t, size=count) / trajectories
        sw = rng.multinomial(seeds, probabilities_s, size=count) / seeds
        ratios = np.zeros((count, seeds), dtype=np.float64)
        for aa, bb in zip(a, b):
            ratios += np.log((tw @ aa.T + EPS) / (tw @ bb.T + EPS)) / len(a)
        samples[start:start + count] = np.sum(ratios * sw, axis=1)
    return point, samples


def _development_bootstrap(family_log_ratios, rng, replicates):
    """Macro family/seed bootstrap; training seeds stay paired across conditions."""
    values = np.asarray(family_log_ratios, dtype=np.float64)
    families, seeds = values.shape
    point = float(values.mean())
    samples = np.empty(replicates)
    for start in range(0, replicates, 2000):
        count = min(2000, replicates - start)
        fw = rng.multinomial(families, np.full(families, 1 / families), size=count) / families
        seed_means = np.empty((count, families))
        for f in range(families):
            sw = rng.multinomial(seeds, np.full(seeds, 1 / seeds), size=count) / seeds
            seed_means[:, f] = sw @ values[f]
        samples[start:start + count] = np.sum(fw * seed_means, axis=1)
    return point, samples


def _effect(point, samples, multiplicity, practical, divergence_ok=True):
    ci = _interval(samples, multiplicity)
    reduction = 100 * (1 - math.exp(point))
    effect_ci = [100 * (1 - math.exp(ci[1])), 100 * (1 - math.exp(ci[0]))]
    if reduction >= practical * 100 and ci[1] < 0 and divergence_ok:
        status = "GREEN"
        conclusion = "supported practical improvement"
    elif reduction <= -practical * 100 and ci[0] > 0:
        status = "RED"
        conclusion = "supported practical harm"
    else:
        status = "YELLOW"
        conclusion = "inconclusive at the prespecified threshold"
    return {"log_mse_ratio": point, "adjusted_log_ci": ci, "reduction_percent": reduction,
            "adjusted_reduction_ci_percent": effect_ci, "ci_level": 1 - .05 / multiplicity,
            "multiplicity": multiplicity, "no_increased_divergence": bool(divergence_ok),
            "status": status, "conclusion": conclusion}


def _mechanisms(cfg, bank, rng):
    rows = []
    multiplicity = len(_families(cfg)) * 4
    conditions = [("correct", "smooth", "iid"), ("correct", "smooth", "response_matched"),
                  ("correct", "smooth", "none"), ("coefficient", "smooth", "none")]
    for family in _families(cfg):
        for prior, arm_a, arm_b in conditions:
            a, b, da, db = [], [], [], []
            for kind in cfg["kinds"]:
                aa, bb = [], []
                for seed in cfg["seeds"]:
                    first = bank[(family, kind, prior, int(seed), arm_a)]["arrays"]["test"]
                    second = bank[(family, kind, prior, int(seed), arm_b)]["arrays"]["test"]
                    aa.append(first["score"])
                    bb.append(second["score"])
                    da.append(np.mean(first["diverged"]))
                    db.append(np.mean(second["diverged"]))
                a.append(aa)
                b.append(bb)
            point, samples = _within_family_bootstrap(a, b, rng, int(cfg.get("bootstrap_replicates", 50000)))
            row = {"family": family, "prior": prior, "candidate": arm_a, "baseline": arm_b,
                   "candidate_divergence_fraction": float(np.mean(da)), "baseline_divergence_fraction": float(np.mean(db))}
            row.update(_effect(point, samples, multiplicity, float(cfg.get("min_practical_reduction", .1)), np.mean(da) <= np.mean(db)))
            rows.append(row)
    iid = [r["family"] for r in rows if r["prior"] == "correct" and r["baseline"] == "iid" and r["status"] == "GREEN"]
    response = [r["family"] for r in rows if r["prior"] == "correct" and r["baseline"] == "response_matched" and r["status"] == "GREEN"]
    ordinary = [r for r in rows if r["baseline"] == "none"]
    status = "GREEN" if len(iid) >= 3 and len(response) >= 2 else "YELLOW"
    return {"status": status, "supported_vs_iid_families": iid, "supported_vs_response_matched_families": response,
            "vs_no_augmentation": {"supported_improvement": sum(r["status"] == "GREEN" for r in ordinary),
                                   "supported_harm": sum(r["status"] == "RED" for r in ordinary),
                                   "inconclusive": sum(r["status"] == "YELLOW" for r in ordinary),
                                   "total_family_prior_contrasts": len(ordinary)},
            "rule": "GREEN requires smooth > IID in at least 3 families and smooth > response-matched IID in at least 2; this does not imply benefit over no augmentation.",
            "contrasts": rows}


def _decision_map(cfg, manifest):
    expected = {(f, k, p, int(s)) for f in _families(cfg) for k in cfg["kinds"] for p in cfg["priors"] for s in cfg["seeds"]}
    decisions, errors = {}, []
    for item in manifest.get("decisions", []):
        try:
            key = _key(item, include_arm=False)
            if key in decisions or key not in expected:
                raise ValueError(f"Duplicate or unexpected selection decision {key}")
            choices = item["choices"]
            for policy in POLICIES:
                if choices.get(policy) not in ARMS:
                    raise ValueError(f"Missing/invalid choice for {policy}")
            if choices["none"] != "none" or choices["always_smooth"] != "smooth":
                raise ValueError("Fixed-policy choices contradict their definitions")
            decisions[key] = item
        except (KeyError, ValueError, TypeError) as exc:
            errors.append(str(exc))
    if expected - set(decisions):
        errors.append(f"Missing {len(expected - set(decisions))} complete selection decisions")
    return decisions, errors


def _isolation(cfg, manifest):
    """Audit declared fit membership without pretending it proves code provenance."""
    families = set(_families(cfg))
    confirmation = cfg.get("confirmation_family", "gray_scott")
    fits = manifest.get("fit_metadata", manifest.get("fits", []))
    if isinstance(fits, dict):
        fits = list(fits.values())
    errors, covered = [], set()
    for fit in fits:
        if not isinstance(fit, dict) or "training_families" not in fit:
            continue
        holdout = fit.get("held_out_family", fit.get("heldout_family"))
        if holdout is None:
            continue
        training = set(fit["training_families"])
        if holdout in training:
            errors.append(f"Selector fit for {holdout} includes its held-out family")
        if confirmation in training:
            errors.append(f"Selector fit for {holdout} includes the independent confirmation family")
        expected = families - {confirmation, holdout}
        if holdout == confirmation:
            expected = families - {confirmation}
        if training != expected:
            errors.append(f"Selector training-family declaration for {holdout} differs from the prespecified development split")
        covered.add(holdout)
    if covered != families:
        errors.append("Selector fit declarations do not cover every development holdout and the independent confirmation family")
    return {"verified_declared_family_isolation": not errors, "errors": errors,
            "meaning": "Audits manifest-declared family membership; test-blind execution also depends on the frozen runner and manifest provenance."}


def _policy_arrays(cfg, bank, decisions, family, policy, baseline):
    a, b, da, db = [], [], [], []
    for kind in cfg["kinds"]:
        for prior in cfg["priors"]:
            aa, bb = [], []
            for seed in cfg["seeds"]:
                key = (family, kind, prior, int(seed))
                choice = decisions[key]["choices"]
                first = bank[key + (choice[policy],)]["arrays"]["test"]
                second = bank[key + (choice[baseline],)]["arrays"]["test"]
                aa.append(first["score"])
                bb.append(second["score"])
                da.append(float(first["diverged"].mean()))
                db.append(float(second["diverged"].mean()))
            a.append(aa)
            b.append(bb)
    return np.asarray(a), np.asarray(b), np.mean(da), np.mean(db)


def _selectors(cfg, bank, decisions, rng):
    confirmation = cfg.get("confirmation_family", "gray_scott")
    development = [f for f in _families(cfg) if f != confirmation]
    rows = []
    for split in ("development_lofo", "confirmation"):
        for baseline in ("none", "best_fixed", "probe_best"):
            divergences_a, divergences_b = [], []
            if split == "confirmation":
                a, b, da, db = _policy_arrays(cfg, bank, decisions, confirmation, "ridge_selector", baseline)
                point, samples = _within_family_bootstrap(a, b, rng, int(cfg.get("bootstrap_replicates", 50000)))
                divergences_a.append(da)
                divergences_b.append(db)
            else:
                family_values = []
                for family in development:
                    a, b, da, db = _policy_arrays(cfg, bank, decisions, family, "ridge_selector", baseline)
                    family_values.append(np.log((a.mean(axis=2) + EPS) / (b.mean(axis=2) + EPS)).mean(axis=0))
                    divergences_a.append(da)
                    divergences_b.append(db)
                point, samples = _development_bootstrap(family_values, rng, int(cfg.get("bootstrap_replicates", 50000)))
            row = {"split": split, "candidate": "ridge_selector", "baseline": baseline,
                   "families": development if split == "development_lofo" else [confirmation],
                   "candidate_divergence_fraction": float(np.mean(divergences_a)), "baseline_divergence_fraction": float(np.mean(divergences_b))}
            row.update(_effect(point, samples, 3, float(cfg.get("min_practical_reduction", .1)), np.mean(divergences_a) <= np.mean(divergences_b)))
            rows.append(row)
    lookup = {(r["split"], r["baseline"]): r for r in rows}
    final_fixed = lookup[("confirmation", "best_fixed")]
    final_probe = lookup[("confirmation", "probe_best")]
    dev_fixed = lookup[("development_lofo", "best_fixed")]
    confirmed_harm = final_fixed["adjusted_log_ci"][0] > 0 or final_probe["adjusted_log_ci"][0] > 0
    if final_fixed["status"] == "GREEN" and not confirmed_harm and dev_fixed["reduction_percent"] > 0:
        status = "GREEN"
    elif confirmed_harm:
        status = "RED"
    else:
        status = "YELLOW"
    return {"status": status, "confirmation_family": confirmation, "development_families": development,
            "rule": "GREEN requires at least 10% confirmed improvement over best fixed, adjusted CI excluding zero, no demonstrated confirmation harm against probe-best, and positive development improvement over best fixed. RED denotes demonstrated confirmation harm against best fixed or probe-best.",
            "uncertainty": f"Only {len(development)} development physical families are resampled; this interval cannot establish generality over all physical systems.",
            "contrasts": rows}


def _costs(cfg, bank, decisions, manifest):
    rows, choices = [], []
    confirmation = cfg.get("confirmation_family", "gray_scott")
    for key, decision in sorted(decisions.items()):
        candidates = {arm: bank[key + (arm,)]["record"] for arm in ARMS}
        probe_cost = sum(float(r.get("probe_setup_seconds", r.get("setup_seconds", 0))) + float(r.get("probe_seconds", 0)) for r in candidates.values())
        all_cost = sum(float(r.get("setup_seconds", 0)) + float(r.get("train_seconds", 0)) for r in candidates.values())
        bank_setup = max(float(r.get("bank_setup_seconds", 0)) for r in candidates.values())
        for policy, arm in decision["choices"].items():
            if policy not in POLICIES:
                continue
            selected = candidates[arm]
            if policy in ("ridge_selector", "probe_best"):
                seconds = probe_cost + float(selected.get("continuation_setup_seconds", 0)) + float(selected.get("remaining_seconds", 0))
            elif policy == "full_validation":
                seconds = all_cost
            else:
                seconds = float(selected.get("setup_seconds", 0)) + float(selected.get("train_seconds", 0))
            if arm != "none" or policy in ("ridge_selector", "probe_best", "full_validation", "hand_rule"):
                seconds += bank_setup
            rows.append({"family": key[0], "kind": key[1], "prior": key[2], "seed": key[3],
                         "split": "confirmation" if key[0] == confirmation else "development_lofo",
                         "policy": policy, "selected_arm": arm, "estimated_training_seconds": seconds,
                         "selected_inference_ms_per_step": selected.get("inference_ms_per_step")})
            choices.append({"family": key[0], "kind": key[1], "prior": key[2], "seed": key[3], "policy": policy, "arm": arm})
    aggregates = []
    for split in ("development_lofo", "confirmation"):
        means = {p: float(np.mean([r["estimated_training_seconds"] for r in rows if r["split"] == split and r["policy"] == p])) for p in POLICIES}
        for policy in POLICIES:
            latency = [r["selected_inference_ms_per_step"] for r in rows if r["split"] == split and r["policy"] == policy and r["selected_inference_ms_per_step"] is not None]
            aggregates.append({"split": split, "policy": policy, "mean_estimated_training_seconds_per_case": means[policy],
                               "cost_ratio_to_none": means[policy] / max(means["none"], EPS),
                               "measured_mean_cost_below_probe_best": means[policy] < means["probe_best"],
                               "mean_selected_inference_ms_per_step": float(np.mean(latency)) if latency else None,
                               "median_selected_inference_ms_per_step": float(np.median(latency)) if latency else None,
                               "selected_inference_measurements": len(latency)})
    shared_bank_times = {}
    for key, value in bank.items():
        bank_key = (key[0], key[2], key[3])
        shared_bank_times[bank_key] = max(shared_bank_times.get(bank_key, 0), float(value["record"].get("bank_setup_seconds", 0)))
    fit_seconds = sum(float(v["record"].get(x, 0)) for v in bank.values() for x in ("setup_seconds", "train_seconds", "evaluation_seconds"))
    development_fit_seconds = sum(float(v["record"].get(x, 0)) for key, v in bank.items() if key[0] != confirmation for x in ("setup_seconds", "train_seconds", "evaluation_seconds"))
    development_bank_seconds = development_fit_seconds + sum(seconds for key, seconds in shared_bank_times.items() if key[0] != confirmation)
    selector_fit_seconds = float(manifest.get("selector_fit_seconds", 0))
    research = {"sum_record_setup_train_evaluation_seconds": fit_seconds,
                "deduplicated_shared_bank_seconds": sum(shared_bank_times.values()),
                "development_bank_seconds": development_bank_seconds,
                "development_bank_plus_selector_fit_seconds": development_bank_seconds + selector_fit_seconds,
                "accounted_research_seconds": fit_seconds + sum(shared_bank_times.values()) + selector_fit_seconds,
                "candidate_fits_executed": len(bank), "selector_fit_seconds": selector_fit_seconds,
                "actual_bank_wall_seconds": manifest.get("bank_wall_seconds"),
                "timing_note": "Shared augmentation-bank/diagnostic setup is charged once per family/prior/seed in research totals, and once per deployment case when a policy needs it. Actual bank wall time, when supplied, is elapsed research time; component timings need not sum exactly to wall time."}
    return {"counterfactual_policy_costs": aggregates, "per_case_costs": rows, "choices": choices,
            "research_bank": research,
            "interpretation": "All candidate fits were executed for the research bank. Policy figures are counterfactual training/selection timing accounts, not actual GPU money saved. Ridge and probe-best pay all candidate probe startup + short probes, then only the selected candidate's continuation startup + remaining training; full-validation pays all complete fits. Legacy records lacking split startup timings allocate setup to the probe. Fixed/random policies pay one full selected fit. Any selected augmentation, and every probe or diagnostic policy, also pays the complete shared augmentation-bank/diagnostic build once. The hand rule is conservatively charged this implemented bundle because a cheaper diagnostic path was not measured. The development-bank subtotal exposes the research cost of obtaining selector training labels; selector fitting is included in the total and disclosed separately. Selected-model inference latency is measured at batch one on validation inputs, including normalization, with warmups excluded and CUDA synchronization. No deployment workload, aggregate inference expense, or energy usage is assumed."}


def _secondary(cfg, bank, decisions):
    horizon = int(cfg.get("primary_horizon", 64))
    end_horizon = int(cfg.get("eval_horizon", horizon))
    cap = float(cfg.get("loss_cap", 1e6))
    model_rows, base_rows, policy_rows = [], [], []
    baseline_seen = set()
    for key, entry in sorted(bank.items()):
        record, arrays = entry["record"], entry["arrays"]
        for split in ("test", "ood"):
            for checkpoint, array_key in (("validation_best", split), ("final_step", f"{split}_final_step")):
                if array_key not in arrays:
                    continue
                for h in sorted({horizon, end_horizon}):
                    if arrays[array_key]["curve"].shape[1] < h:
                        continue
                    score, div, clipped = _score(arrays[array_key]["curve"], h, cap)
                    model_rows.append({"family": key[0], "kind": key[1], "prior": key[2], "seed": key[3], "arm": key[4],
                                       "checkpoint": checkpoint, "split": split, "horizon": h, "mean_capped_mse": float(score.mean()),
                                       "diverged_trajectories": int(div.sum()), "capped_trajectories": int(clipped.sum()),
                                       "trajectories": len(score), "parameter_count": record.get("parameter_count"),
                                       "inference_ms_per_step": record.get("inference_ms_per_step") if checkpoint == "validation_best" else None})
            for baseline in ("persistence", "mechanistic"):
                basekey = (key[0], key[2], split, baseline)
                if f"{split}_{baseline}" not in arrays or basekey in baseline_seen:
                    continue
                baseline_seen.add(basekey)
                score, div, clipped = arrays[f"{split}_{baseline}"]
                base_rows.append({"family": key[0], "prior": key[2], "split": split, "baseline": baseline,
                                  "horizon": horizon, "mean_capped_mse": float(score.mean()),
                                  "diverged_trajectories": int(div.sum()), "capped_trajectories": int(clipped.sum()), "trajectories": len(score)})
    for key, decision in sorted(decisions.items()):
        candidate_means = {a: float(bank[key + (a,)]["arrays"]["test"]["score"].mean()) for a in ARMS}
        oracle = min(candidate_means.values())
        for policy, arm in decision["choices"].items():
            if policy not in POLICIES:
                continue
            arrays = bank[key + (arm,)]["arrays"]
            for split in ("test", "ood"):
                score = arrays[split]["score"]
                row = {"family": key[0], "kind": key[1], "prior": key[2], "seed": key[3], "policy": policy, "arm": arm,
                       "checkpoint": "validation_best", "split": split, "horizon": horizon, "mean_capped_mse": float(score.mean()),
                       "diverged_trajectories": int(arrays[split]["diverged"].sum()), "trajectories": len(score)}
                if split == "test":
                    row["regret_log_ratio_to_test_oracle"] = float(math.log((score.mean() + EPS) / (oracle + EPS)))
                policy_rows.append(row)
    return {"models": model_rows, "physics_and_persistence": base_rows, "policies": policy_rows,
            "note": "OOD, long-horizon, and exact final-training-step network outcomes are descriptive secondary results. Primary comparisons and all policies use validation-best checkpoints. The NPZ exact_mse arrays contain final-step networks, not exact physics. The test oracle is an inaccessible hindsight reference used only to measure regret; it never chooses deployed policies. Architectures are not parameter- or compute-matched."}


def _save(output, report):
    directory = output / "reports"
    directory.mkdir(parents=True, exist_ok=True)
    report = _jsonable(report)
    (directory / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for name, rows in (("mechanism_contrasts", report.get("mechanism", {}).get("contrasts", [])),
                       ("selector_contrasts", report.get("selector", {}).get("contrasts", [])),
                       ("policy_costs", report.get("costs", {}).get("per_case_costs", [])),
                       ("policy_cost_summary", report.get("costs", {}).get("counterfactual_policy_costs", [])),
                       ("model_metrics", report.get("secondary", {}).get("models", [])),
                       ("policy_metrics", report.get("secondary", {}).get("policies", [])),
                       ("physics_baselines", report.get("secondary", {}).get("physics_and_persistence", []))):
        _write_csv(directory / f"{name}.csv", rows)
    lines = ["# Physical-family mechanism and selector results", "",
             f"- Completion: **{report['completion']['status']}** ({report['completion']['valid_records']}/{report['completion']['expected_records']} candidate records)",
             f"- Scientific eligibility: **{report['scientific_status']}**",
             f"- Mechanism: **{report.get('mechanism', {}).get('status', 'NOT_EVALUATED')}**",
             f"- Selector: **{report.get('selector', {}).get('status', 'NOT_EVALUATED')}**", ""]
    if report.get("errors"):
        lines += ["## Issues", ""] + [f"- {e}" for e in report["errors"]] + [""]
    if report.get("mechanism", {}).get("contrasts"):
        lines += ["## Prespecified mechanism contrasts", "", "Positive reduction favors smooth augmentation. CIs are Bonferroni-adjusted within the mechanism family.", "",
                  "| Physical family | Prior | Baseline | Reduction % | Adjusted CI % | Result |", "|---|---|---|---:|---|---|"]
        for row in report["mechanism"]["contrasts"]:
            lo, hi = row["adjusted_reduction_ci_percent"]
            lines.append(f"| {row['family']} | {row['prior']} | {row['baseline']} | {row['reduction_percent']:.1f} | [{lo:.1f}, {hi:.1f}] | {row['status']} |")
        lines += ["", report["mechanism"]["rule"], "", "## Prespecified selector contrasts", "",
                  "| Split | Baseline | Reduction % | Adjusted CI % | Result |", "|---|---|---:|---|---|"]
        for row in report["selector"]["contrasts"]:
            lo, hi = row["adjusted_reduction_ci_percent"]
            lines.append(f"| {row['split']} | {row['baseline']} | {row['reduction_percent']:.1f} | [{lo:.1f}, {hi:.1f}] | {row['status']} |")
        lines += ["", report["selector"]["rule"], "", report["selector"]["uncertainty"], "", report["costs"]["interpretation"], ""]
    lines += ["## Interpretation limits", ""] + [f"- {v}" for v in report.get("interpretation", [])] + [""]
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def analyze(output, cfg, records, manifest, eligible=True):
    """Validate evidence, compute fixed contrasts, and write reusable reports."""
    output = Path(output)
    bank, errors, expected = _load(output, cfg, records)
    decisions, decision_errors = _decision_map(cfg, manifest)
    errors.extend(decision_errors)
    isolation = _isolation(cfg, manifest)
    profile = cfg.get("profile", "unknown")
    complete = not errors
    science = complete and bool(eligible) and profile == "full" and isolation["verified_declared_family_isolation"]
    report = {"schema_version": 1, "profile": profile,
              "completion": {"status": "GREEN" if complete else "RED", "valid_records": len(bank), "expected_records": expected},
              "scientific_status": "EVALUATED" if science else "NOT_EVALUATED", "errors": errors,
              "holdout_isolation": isolation, "mechanism": {"status": "NOT_EVALUATED"}, "selector": {"status": "NOT_EVALUATED"},
              "interpretation": ["A successful completed run can yield positive, inconclusive, or negative scientific findings.",
                                 "Mechanism support against IID/response matching does not establish an improvement over ordinary supervised training.",
                                 "Pure physics and persistence baselines are reported; a correct simulator may outperform every learned model.",
                                 "This experiment selects augmentation within each architecture, not a winner among disjoint transformer/physics/world-model categories.",
                                 "Matched update counts are not matched compute or parameter counts.",
                                 "Scores cap each trajectory's mean MSE at the configured loss cap; failed trajectories remain in the analysis. Log ratios use an additive 1e-15 floor.",
                                 "The selection manifest must be frozen before test evaluation. Reports audit declared family membership and evidence hashes, not an external timestamp authority."]}
    if not science:
        report["scientific_ineligibility_reasons"] = (["Full profile required for scientific gates"] if profile != "full" else []) + (["Runner marked this evidence ineligible"] if not eligible else []) + errors + isolation["errors"]
    if complete:
        report["costs"] = _costs(cfg, bank, decisions, manifest)
        report["secondary"] = _secondary(cfg, bank, decisions)
        if science:
            rng = np.random.default_rng(int(cfg.get("bootstrap_seed", 260912)))
            report["mechanism"] = _mechanisms(cfg, bank, rng)
            report["selector"] = _selectors(cfg, bank, decisions, rng)
        else:
            report["mechanism"]["reason"] = "Scientific gates are disabled for smoke/pilot, incomplete, or unaudited runs. Descriptive metrics and timings remain available."
            report["selector"]["reason"] = report["mechanism"]["reason"]
    return _save(output, report)


def print_console(report, color="auto"):
    """Print terminal-safe colors plus explicit words for redirected logs/Jupyter."""
    use_color = color in ("always", True) or (color == "auto" and sys.stdout.isatty() and "NO_COLOR" not in os.environ)
    codes = {"GREEN": "32", "YELLOW": "33", "RED": "31", "NOT_EVALUATED": "36", "EVALUATED": "36"}
    def tag(status):
        return f"\033[{codes.get(status, '0')}m{status}\033[0m" if use_color else status
    print("\nPHYSICAL-FAMILY EXPERIMENT RESULTS")
    completion = report["completion"]
    print(f"Completion: {tag(completion['status'])} | {completion['valid_records']}/{completion['expected_records']} candidate records verified")
    print(f"Scientific eligibility: {tag(report['scientific_status'])}")
    if report["scientific_status"] != "EVALUATED":
        print(f"Mechanism: {tag('NOT_EVALUATED')} | Selector: {tag('NOT_EVALUATED')}")
        for error in report.get("scientific_ineligibility_reasons", [])[:8]:
            print(f"  - {error}")
        print("Read reports/report.md and reports/report.json for diagnostics and descriptive metrics.")
        return
    mechanism = report["mechanism"]
    print(f"Mechanism: {tag(mechanism['status'])}")
    print(f"  Smooth beats IID with adjusted support in {len(mechanism['supported_vs_iid_families'])} families; response-matched IID in {len(mechanism['supported_vs_response_matched_families'])}.")
    ordinary = mechanism["vs_no_augmentation"]
    print(f"  Versus no augmentation: {ordinary['supported_improvement']} supported improvements, {ordinary['supported_harm']} supported harms, {ordinary['inconclusive']} inconclusive family/prior contrasts.")
    for row in mechanism["contrasts"]:
        lo, hi = row["adjusted_reduction_ci_percent"]
        print(f"  {row['family']:18s} {row['prior']:11s} smooth vs {row['baseline']:16s}: {row['reduction_percent']:+7.1f}% [{lo:+.1f}, {hi:+.1f}] {tag(row['status'])}")
    selector = report["selector"]
    print(f"Selector: {tag(selector['status'])} | confirmation family: {selector['confirmation_family']}")
    for row in selector["contrasts"]:
        lo, hi = row["adjusted_reduction_ci_percent"]
        print(f"  {row['split']:18s} ridge vs {row['baseline']:11s}: {row['reduction_percent']:+7.1f}% [{lo:+.1f}, {hi:+.1f}] {tag(row['status'])}")
    print("  Positive percentages are reductions in the geometric mean paired MSE ratio.")
    print("Selection cost (counterfactual; every research candidate was actually trained):")
    for row in report["costs"]["counterfactual_policy_costs"]:
        if row["split"] == "confirmation":
            latency = row.get("mean_selected_inference_ms_per_step")
            latency_text = f", batch-one inference {latency:.3f} ms/step" if latency is not None else ""
            print(f"  {row['policy']:16s}: {row['mean_estimated_training_seconds_per_case']:.1f} s/case, {row['cost_ratio_to_none']:.2f}x ordinary training{latency_text}")
    print("  Latency is descriptive; no deployment workload, energy, or actual GPU-money-saving claim is made.")
    print("Detailed curves, divergences, physics baselines, selection regret and costs: reports/report.json and reports/*.csv")
    print("Readable report: reports/report.md\n")
