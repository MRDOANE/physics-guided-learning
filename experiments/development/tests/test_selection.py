"""Selection gates: family leakage, candidate completeness, and acquisition cost."""
import copy
import json
import math

import numpy as np
import pytest

from lpb.selection import ARMS, FEATURES, feature_names, freeze_choices, manifest_digest


def bank():
    families = ("wave", "burgers", "ks", "gray_scott")
    records = []
    for family_index, family in enumerate(families):
        for seed in (11, 22):
            task_id = f"{family}__transformer__correct__{seed}"
            diagnostics = dict(gap_relative=0.08 + family_index * 0.02, gain_iid=3.0,
                               gain_smooth=0.8 + seed / 100, low_power=0.8,
                               high_power=0.2, increment_rms=0.03)
            for arm, probe, final in zip(ARMS, (1.0, 1.2, 0.6, 0.8), (1.0, 1.4, 0.5, 0.9)):
                record = dict(task_id=task_id, family=family, kind="transformer", prior="correct",
                              seed=seed, arm=arm, features=diagnostics.copy(), probe_val=probe,
                              probe_seconds=2.0, remaining_seconds=8.0, train_seconds=10.0,
                              setup_seconds=1.0, diagnostic_seconds=0.5)
                if family != "gray_scott":
                    record["final_val"] = final
                records.append(record)
    cfg = {"selector": {"development_families": ["wave", "burgers", "ks"],
                        "confirmation_family": "gray_scott"}}
    return records, cfg


def decisions_for(manifest, family):
    return [decision for decision in manifest["decisions"] if decision["family"] == family]


def without_timing(manifest):
    result = copy.deepcopy(manifest)
    result.pop("selector_fit_seconds")
    for model in result["models"].values():
        model.pop("fit_seconds")
    for fit in result["fit_metadata"]:
        fit.pop("fit_seconds")
    return result


def test_heldout_final_validation_cannot_change_its_predictions_or_choices():
    records, cfg = bank()
    original = freeze_choices(records, cfg)
    for record in records:
        if record["family"] == "wave":
            record["final_val"] = 1e-12 if record["arm"] == "iid" else 1e6
    changed = freeze_choices(records, cfg)
    assert decisions_for(original, "wave") == decisions_for(changed, "wave")
    original_model = {key: value for key, value in original["models"]["heldout_wave"].items() if key != "fit_seconds"}
    changed_model = {key: value for key, value in changed["models"]["heldout_wave"].items() if key != "fit_seconds"}
    assert original_model == changed_model


def test_confirmation_final_validation_is_entirely_ignored():
    records, cfg = bank()
    original = freeze_choices(records, cfg)
    for record in records:
        if record["family"] == "gray_scott":
            record["final_val"] = -999 if record["arm"] == "none" else float("nan")
    changed = freeze_choices(records, cfg)
    assert without_timing(original) == without_timing(changed)


def test_outer_and_inner_families_are_disjoint_and_scaling_is_local():
    records, cfg = bank()
    first = freeze_choices(records, cfg)
    for record in records:
        if record["family"] == "wave":
            record["features"]["gap_relative"] = 99999.0
    second = freeze_choices(records, cfg)
    model = second["models"]["heldout_wave"]
    assert model["mean"] == first["models"]["heldout_wave"]["mean"]
    assert model["scale"] == first["models"]["heldout_wave"]["scale"]
    assert np.isclose(model["mean"][0], 0.11)
    for fit in second["fit_metadata"]:
        assert fit["held_out_family"] not in fit["training_families"]
        for inner in fit["inner_cv"]["folds"]:
            assert inner["held_out_family"] not in inner["training_families"]
            assert fit["held_out_family"] not in inner["training_families"]
    assert second["models"]["locked_confirmation"]["training_families"] == ["burgers", "ks", "wave"]
    assert not any(name in feature_names() for name in ("family", "prior", "seed"))


def test_choices_are_deterministic_under_record_order_and_ties():
    records, cfg = bank()
    for record in records:
        record["probe_val"] = 1.0
        if record["family"] != "gray_scott":
            record["final_val"] = 1.0
    first = freeze_choices(records, cfg)
    second = freeze_choices(list(reversed(records)), cfg)
    assert without_timing(first) == without_timing(second)
    for decision in first["decisions"]:
        assert decision["choices"]["ridge_selector"] == "none"
        assert decision["choices"]["probe_best"] == "none"
        assert decision["choices"]["best_fixed"] == "none"


def test_abstention_requires_five_percent_predicted_gain():
    records, cfg = bank()
    for record in records:
        if record["family"] != "gray_scott":
            record["final_val"] = 1.0 if record["arm"] == "none" else 0.98
    result = freeze_choices(records, cfg)
    assert all(decision["choices"]["ridge_selector"] == "none" for decision in result["decisions"])


def test_positive_signal_selects_smooth_and_costs_charge_all_probes():
    records, cfg = bank()
    result = freeze_choices(records, cfg)
    for decision in result["decisions"]:
        assert decision["choices"]["ridge_selector"] == "smooth"
        assert decision["choices"]["hand_rule"] == "smooth"
        costs = decision["costs"]
        assert costs["ridge_selector"]["probe_arms"] == list(ARMS)
        assert costs["ridge_selector"]["total_target_seconds"] == 20.5
        assert costs["probe_best"]["total_target_seconds"] == 20.5
        assert costs["none"]["total_target_seconds"] == 11.0
        assert costs["hand_rule"]["total_target_seconds"] == 11.5


def test_best_fixed_is_architecture_specific_and_excludes_target_labels():
    records, cfg = bank()
    fno_records = copy.deepcopy(records)
    fno_losses = {"none": 1.0, "iid": 0.3, "smooth": 2.0, "response_matched": 1.1}
    for record in fno_records:
        record["kind"] = "fno"
        record["task_id"] = record["task_id"].replace("transformer", "fno")
        if record["family"] != "gray_scott":
            record["final_val"] = fno_losses[record["arm"]]
    records.extend(fno_records)
    first = freeze_choices(records, cfg)
    for decision in first["decisions"]:
        expected = "smooth" if decision["kind"] == "transformer" else "iid"
        assert decision["choices"]["best_fixed"] == expected
    for model in first["models"].values():
        assert model["best_fixed_by_kind"] == {"transformer": "smooth", "fno": "iid"}
        assert set(model["best_fixed_scores_by_kind"]) == {"transformer", "fno"}
    for record in records:
        if record["family"] == "wave":
            record["final_val"] = 1e-12 if record["arm"] == "response_matched" else 1e6
    second = freeze_choices(records, cfg)
    assert decisions_for(first, "wave") == decisions_for(second, "wave")
    assert first["models"]["heldout_wave"]["best_fixed_by_kind"] == second["models"]["heldout_wave"]["best_fixed_by_kind"]


def test_failures_are_capped_without_dropping_candidates():
    records, cfg = bank()
    for record in records:
        if record["arm"] == "iid":
            record["probe_val"] = float("nan")
            if record["family"] != "gray_scott":
                record["final_val"] = float("inf")
    result = freeze_choices(records, cfg)
    assert len(result["decisions"]) == 8
    assert all(math.isfinite(score) for decision in result["decisions"]
               for score in decision["predicted_log_relative_loss"].values())
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("damage", ["missing_arm", "missing_family", "missing_label", "duplicate", "features", "test_field"])
def test_incomplete_or_leaky_inputs_fail_closed(damage):
    records, cfg = bank()
    if damage == "missing_arm":
        records.pop()
    elif damage == "missing_family":
        records = [record for record in records if record["family"] != "ks"]
    elif damage == "missing_label":
        records[0].pop("final_val")
    elif damage == "duplicate":
        records.append(copy.deepcopy(records[0]))
    elif damage == "features":
        records[0]["features"]["prior"] = 0
    else:
        records[0]["test_mse"] = 0.001
    with pytest.raises(ValueError):
        freeze_choices(records, cfg)


def test_config_inventory_detects_whole_missing_task():
    records, cfg = bank()
    cfg.update(kinds=["transformer"], priors=["correct"], seeds=[11, 22])
    freeze_choices(records, cfg)
    records = [record for record in records if not (record["family"] == "wave" and record["seed"] == 22)]
    with pytest.raises(ValueError, match="configured schedule"):
        freeze_choices(records, cfg)


def test_unknown_confirmation_remaining_cost_is_not_claimed_as_zero():
    records, cfg = bank()
    for record in records:
        if record["family"] == "gray_scott":
            record["remaining_seconds"] = None
            record["train_seconds"] = None
    result = freeze_choices(records, cfg)
    for decision in decisions_for(result, "gray_scott"):
        assert decision["costs"]["ridge_selector"]["total_target_seconds"] is None
        assert decision["costs"]["none"]["total_target_seconds"] is None
    assert len(manifest_digest(result)) == 64


def test_smoke_two_family_fallback_is_explicit():
    records, cfg = bank()
    records = [record for record in records if record["family"] != "ks"]
    cfg["selector"]["development_families"] = ["wave", "burgers"]
    result = freeze_choices(records, cfg)
    model = result["models"]["heldout_wave"]
    assert model["alpha"] == 10
    assert model["inner_cv"]["mode"] == "smoke_only_fixed_alpha_insufficient_inner_families"
