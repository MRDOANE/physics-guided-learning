"""Numerical validation of equations, reference fidelity, and training labels."""
import math
import unittest

import numpy as np
import torch

from lpb.physics import (approx_params, initial_states, make_actions,
                         numerical_audit, sample_params, simulate, spec, step)


FAMILIES = ("wave", "burgers", "ks", "advection_diffusion", "allen_cahn", "gray_scott")


class PhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_every_family_reference_converges_and_respects_invariants(self):
        for name in FAMILIES:
            with self.subTest(family=name):
                audit = numerical_audit(name)
                self.assertTrue(audit["passed"], audit)

    def test_full_declared_horizon_is_finite_for_id_and_ood(self):
        """The audit's short window alone cannot detect late nonlinear failures."""
        for name in FAMILIES:
            with self.subTest(family=name):
                rng = np.random.default_rng(823459)
                params = np.concatenate((sample_params(name, 4, rng),
                                         sample_params(name, 4, rng, True)))
                states = simulate(name, params, initial_states(name, 8, 32, rng),
                                  make_actions(name, 8, 184, rng), "fine")
                self.assertTrue(np.isfinite(states).all())
                self.assertGreater(np.std(states[:, 24:]), 1e-3)
                self.assertGreater(np.sqrt(np.mean(np.diff(states[:, -32:], axis=1) ** 2)), 1e-5)
                if name == "gray_scott":
                    self.assertGreaterEqual(float(states.min()), -1e-9)
                    # Each split should retain nontrivial product concentration.
                    self.assertGreater(float(states[:4, -32:, 1].mean()), 1e-3)
                    self.assertGreater(float(states[4:, -32:, 1].mean()), 1e-3)

    def test_coarse_physics_labels_have_finite_input_and_parameter_gradients(self):
        for name in FAMILIES:
            rng = np.random.default_rng(647284)
            p = np.concatenate((sample_params(name, 3, rng), sample_params(name, 3, rng, True)))
            initial = initial_states(name, 6, 32, rng)
            actions = make_actions(name, 6, 25, rng)
            evolved = simulate(name, p, initial, actions, "fine")[:, -1]
            scale = np.maximum(evolved.std(axis=(0, 2), keepdims=True), 1e-3)
            perturbed = evolved + rng.normal(size=evolved.shape) * scale * 0.05
            for mismatch in ("correct", "coefficient"):
                with self.subTest(family=name, mismatch=mismatch):
                    y = torch.tensor(perturbed, dtype=torch.float32, requires_grad=True)
                    params = torch.tensor(p, dtype=torch.float32, requires_grad=True)
                    action = torch.tensor(actions[:, -1], dtype=torch.float32, requires_grad=True)
                    predicted = step(name, y, action, approx_params(name, params, mismatch), mismatch)
                    self.assertTrue(torch.isfinite(predicted).all())
                    predicted.square().mean().backward()
                    for grad in (y.grad, params.grad, action.grad):
                        self.assertIsNotNone(grad)
                        self.assertTrue(torch.isfinite(grad).all())

    def test_advection_diffusion_matches_exact_translated_fourier_mode(self):
        x = torch.arange(32, dtype=torch.float64) * (2 * math.pi / 32)
        p = torch.tensor([[0.06, 1.2, 0.04]], dtype=torch.float64)
        y = torch.cos(3 * x)[None, None]
        zero = torch.zeros((1, 1), dtype=torch.float64)
        dt = spec("advection_diffusion")["dt"]
        expected = math.exp(-(0.06 * 9 + 0.04) * dt) * torch.cos(3 * (x - 1.2 * dt))
        actual = step("advection_diffusion", y, zero, p)
        torch.testing.assert_close(actual[0, 0], expected, rtol=1e-12, atol=1e-12)

    def test_advection_diffusion_nyquist_damps_without_spurious_rotation(self):
        y = ((-1.) ** torch.arange(32, dtype=torch.float64))[None, None]
        p = torch.tensor([[0.06, 1.2, 0.04]], dtype=torch.float64)
        actual = step("advection_diffusion", y, torch.zeros((1, 1), dtype=torch.float64), p)
        expected = y * math.exp(-(0.06 * 16 ** 2 + 0.04) * spec("advection_diffusion")["dt"])
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_allen_cahn_homogeneous_reaction_matches_analytic_solution(self):
        value, reaction = 0.3, 1.2
        y = torch.full((1, 1, 32), value, dtype=torch.float64)
        p = torch.tensor([[0.035, reaction, 0.2]], dtype=torch.float64)
        actual = step("allen_cahn", y, torch.zeros((1, 1), dtype=torch.float64), p, fidelity="check")
        dt = spec("allen_cahn")["dt"]
        expected = value / math.sqrt(value ** 2 + (1 - value ** 2) * math.exp(-2 * reaction * dt))
        torch.testing.assert_close(actual, torch.full_like(actual, expected), rtol=1e-8, atol=1e-10)

    def test_gray_scott_reaction_only_conserves_substrate_plus_product(self):
        rng = np.random.default_rng(117)
        y = torch.tensor(initial_states("gray_scott", 4, 32, rng), dtype=torch.float64)
        zero_params = torch.zeros((4, 3), dtype=torch.float64)
        actual = step("gray_scott", y, torch.zeros((4, 1), dtype=torch.float64), zero_params)
        torch.testing.assert_close(actual.sum(1), y.sum(1), rtol=1e-12, atol=1e-12)

    def test_gray_scott_product_free_feed_matches_analytic_solution(self):
        y = torch.zeros((1, 2, 32), dtype=torch.float64)
        y[:, 0] = 0.8
        p = torch.tensor([[0.16, 0.03, 0.04]], dtype=torch.float64)
        actual = step("gray_scott", y, torch.zeros((1, 1), dtype=torch.float64), p, fidelity="check")
        expected = 1 - 0.2 * math.exp(-0.03 * spec("gray_scott")["dt"])
        torch.testing.assert_close(actual[:, 0], torch.full_like(actual[:, 0], expected), rtol=1e-11, atol=1e-12)
        torch.testing.assert_close(actual[:, 1], torch.zeros_like(actual[:, 1]), rtol=0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
