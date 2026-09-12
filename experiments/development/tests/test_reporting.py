"""Synthetic tests of pairing, failed trajectories, holdout gates, and cost logic."""
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


SPEC = importlib.util.spec_from_file_location("reporting_under_test", Path(__file__).parents[1] / "lpb" / "reporting.py")
reporting = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reporting)


def fixture(root, profile="full"):
    families = ["wave", "burgers", "reaction", "advection", "ks", "gray_scott"]
    cfg = {"profile": profile, "families": families, "kinds": ["transformer", "looped"],
           "priors": ["correct", "coefficient"], "seeds": [101, 202, 303],
           "confirmation_family": "gray_scott", "primary_horizon": 2, "eval_horizon": 3,
           "bootstrap_replicates": 400, "bootstrap_seed": 260912, "loss_cap": 1e6,
           "min_practical_reduction": .1}
    records, decisions = [], []
    (root / "evaluations").mkdir()
    for family in families:
        for kind in cfg["kinds"]:
            for prior in cfg["priors"]:
                for seed in cfg["seeds"]:
                    case = {"family": family, "kind": kind, "prior": prior, "seed": seed}
                    choices = {"ridge_selector": "smooth", "none": "none", "always_smooth": "smooth",
                               "best_fixed": "none", "random": "iid", "hand_rule": "smooth",
                               "probe_best": "smooth", "full_validation": "smooth"}
                    decisions.append({**case, "choices": choices})
                    for arm, multiplier in {"none": 1., "iid": 2., "smooth": .5, "response_matched": 1.5}.items():
                        task = f"{family}_{kind}_{prior}_{seed}_{arm}"
                        path = root / "evaluations" / f"{task}.npz"
                        trajectory = np.array([1., 2., 3., 4.]) * (1 + seed / 10000)
                        curve = np.repeat(trajectory[:, None], 3, axis=1) * multiplier
                        np.savez(path, test_mse=curve, ood_mse=curve * 2,
                                 test_exact_mse=curve * 3, ood_exact_mse=curve * 6,
                                 test_ids=np.array(["a", "b", "c", "d"]), ood_ids=np.array(["e", "f", "g", "h"]),
                                 test_persistence_mse=curve * 10, ood_persistence_mse=curve * 20,
                                 test_mechanistic_mse=curve * .001, ood_mechanistic_mse=curve * .002)
                        records.append({**case, "arm": arm, "task_id": task, "final_val": multiplier,
                                        "probe_val": multiplier * 1.1, "setup_seconds": 1., "probe_seconds": 2.,
                                        "remaining_seconds": 8., "train_seconds": 10., "evaluation_seconds": .1,
                                        "bank_setup_seconds": 5., "diagnostic_seconds": 5., "parameter_count": 100,
                                        "inference_ms_per_step": multiplier / 10,
                                        "evaluation_path": str(path.relative_to(root)),
                                        "evaluation_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    fits = [{"held_out_family": f, "training_families": [g for g in families if g not in {f, "gray_scott"}]} for f in families]
    manifest = {"decisions": decisions, "fit_metadata": fits, "selector_fit_seconds": .25}
    return cfg, records, manifest


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_crossed_bootstrap_is_paired_and_scale_invariant(self):
        a = np.arange(1, 1 + 2 * 3 * 5, dtype=float).reshape(2, 3, 5)
        point, draws = reporting._within_family_bootstrap(a, a * 2, np.random.default_rng(1), 600)
        self.assertAlmostEqual(point, -np.log(2))
        np.testing.assert_allclose(draws, -np.log(2), atol=1e-12)

    def test_cap_is_applied_after_time_mean_and_divergences_are_kept(self):
        scores, divergence, capped = reporting._score([[0, 200], [np.inf, 0], [np.nan, 1]], 2, 100)
        np.testing.assert_equal(scores, [100, 100, 100])
        np.testing.assert_equal(divergence, [False, True, True])
        np.testing.assert_equal(capped, [False, True, True])
        with self.assertRaisesRegex(ValueError, "negative"):
            reporting._score([[1, -1]], 2, 100)

    def test_full_run_separates_mechanism_selector_and_research_cost(self):
        cfg, records, manifest = fixture(self.root)
        result = reporting.analyze(self.root, cfg, records, manifest)
        self.assertEqual(result["completion"]["status"], "GREEN")
        self.assertEqual(result["scientific_status"], "EVALUATED")
        self.assertEqual(result["mechanism"]["status"], "GREEN")
        self.assertEqual(result["selector"]["status"], "GREEN")
        self.assertEqual(len(result["mechanism"]["contrasts"]), 24)
        self.assertEqual(len(result["selector"]["contrasts"]), 6)
        self.assertEqual(result["mechanism"]["vs_no_augmentation"]["supported_improvement"], 12)
        costs = {r["policy"]: r for r in result["costs"]["counterfactual_policy_costs"] if r["split"] == "confirmation"}
        self.assertEqual(costs["none"]["mean_estimated_training_seconds_per_case"], 11)
        self.assertEqual(costs["ridge_selector"]["mean_estimated_training_seconds_per_case"], 25)
        self.assertEqual(costs["full_validation"]["mean_estimated_training_seconds_per_case"], 49)
        self.assertEqual(costs["hand_rule"]["mean_estimated_training_seconds_per_case"], 16)
        self.assertAlmostEqual(costs["ridge_selector"]["mean_selected_inference_ms_per_step"], .05)
        self.assertEqual(result["costs"]["research_bank"]["deduplicated_shared_bank_seconds"], 6 * 2 * 3 * 5)
        self.assertTrue((self.root / "reports" / "report.md").is_file())
        self.assertTrue((self.root / "reports" / "policy_metrics.csv").is_file())

    def test_final_step_networks_are_not_deduplicated_into_physics_baselines(self):
        cfg, records, manifest = fixture(self.root, profile="smoke")
        result = reporting.analyze(self.root, cfg, records, manifest)
        models = result["secondary"]["models"]
        best = [r for r in models if r["checkpoint"] == "validation_best"]
        final = [r for r in models if r["checkpoint"] == "final_step"]
        self.assertEqual(len(best), len(records) * 2 * 2)
        self.assertEqual(len(final), len(best))
        model_key = lambda r: (r["family"], r["kind"], r["prior"], r["seed"], r["arm"], r["split"], r["horizon"])
        best_by_key = {model_key(row): row for row in best}
        for row in final:
            self.assertAlmostEqual(row["mean_capped_mse"], 3 * best_by_key[model_key(row)]["mean_capped_mse"])
        self.assertEqual({r["baseline"] for r in result["secondary"]["physics_and_persistence"]}, {"persistence", "mechanistic"})

    def test_selector_pays_only_selected_continuation_startup(self):
        cfg, records, manifest = fixture(self.root, profile="smoke")
        for record in records:
            record["probe_setup_seconds"] = .25
            record["continuation_setup_seconds"] = .75
        result = reporting.analyze(self.root, cfg, records, manifest)
        costs = {r["policy"]: r for r in result["costs"]["counterfactual_policy_costs"] if r["split"] == "confirmation"}
        self.assertEqual(costs["ridge_selector"]["mean_estimated_training_seconds_per_case"], 4 * (.25 + 2) + .75 + 8 + 5)
        self.assertEqual(costs["probe_best"]["mean_estimated_training_seconds_per_case"], 22.75)
        self.assertEqual(costs["none"]["mean_estimated_training_seconds_per_case"], 11)
        self.assertEqual(costs["full_validation"]["mean_estimated_training_seconds_per_case"], 49)
        research = result["costs"]["research_bank"]
        self.assertAlmostEqual(research["accounted_research_seconds"], len(records) * 11.1 + 6 * 2 * 3 * 5 + .25)
        self.assertAlmostEqual(research["development_bank_seconds"], (len(records) * 5 / 6) * 11.1 + 5 * 2 * 3 * 5)
        self.assertAlmostEqual(research["development_bank_plus_selector_fit_seconds"], research["development_bank_seconds"] + .25)

    def test_smoke_never_receives_scientific_green(self):
        cfg, records, manifest = fixture(self.root, profile="smoke")
        result = reporting.analyze(self.root, cfg, records, manifest)
        self.assertEqual(result["completion"]["status"], "GREEN")
        self.assertEqual(result["scientific_status"], "NOT_EVALUATED")
        self.assertEqual(result["mechanism"]["status"], "NOT_EVALUATED")
        self.assertEqual(result["selector"]["status"], "NOT_EVALUATED")

    def test_missing_and_tampered_records_disable_science(self):
        cfg, records, manifest = fixture(self.root)
        (self.root / records[0]["evaluation_path"]).write_bytes(b"tampered")
        result = reporting.analyze(self.root, cfg, records[:-1], manifest)
        self.assertEqual(result["completion"]["status"], "RED")
        self.assertEqual(result["scientific_status"], "NOT_EVALUATED")
        self.assertTrue(any("SHA256 mismatch" in v for v in result["errors"]))
        self.assertEqual(result["completion"]["valid_records"], len(records) - 2)

    def test_confirmation_family_leakage_disables_gate(self):
        cfg, records, manifest = fixture(self.root)
        manifest["fit_metadata"][0]["training_families"].append("gray_scott")
        result = reporting.analyze(self.root, cfg, records, manifest)
        self.assertEqual(result["scientific_status"], "NOT_EVALUATED")
        self.assertFalse(result["holdout_isolation"]["verified_declared_family_isolation"])

    def test_confirmation_decisions_do_not_change_development_contrasts(self):
        cfg, records, manifest = fixture(self.root)
        first = reporting.analyze(self.root, cfg, records, manifest)
        for decision in manifest["decisions"]:
            if decision["family"] == "gray_scott":
                decision["choices"]["ridge_selector"] = "iid"
        second = reporting.analyze(self.root, cfg, records, manifest)
        self.assertEqual([r for r in first["selector"]["contrasts"] if r["split"] == "development_lofo"],
                         [r for r in second["selector"]["contrasts"] if r["split"] == "development_lofo"])
        self.assertEqual(second["selector"]["status"], "RED")

    def test_permuted_trajectory_rows_are_realigned_by_ids(self):
        cfg, records, manifest = fixture(self.root)
        record = records[0]
        path = self.root / record["evaluation_path"]
        with np.load(path) as bundle:
            arrays = {key: value[::-1].copy() for key, value in bundle.items()}
        np.savez(path, **arrays)
        record["evaluation_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        result = reporting.analyze(self.root, cfg, records, manifest)
        self.assertEqual(result["completion"]["status"], "GREEN")
        self.assertEqual(result["mechanism"]["status"], "GREEN")


if __name__ == "__main__":
    unittest.main()
