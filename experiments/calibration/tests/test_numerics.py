"""Independent numerical checks against archived equations and dataset hashes.

Run from the package with:
    python -m unittest discover -s tests -p test_numerics.py -v

PHYSICS_BASELINE_PACKAGE may point to the package containing assets/. Optional
PHYSICS_DEVELOPMENT_SOURCE points to the historical development physics.py for
cross-release checks. No neural model is trained by this test suite.
"""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import unittest

import numpy as np

from data_backend import create_backend, dataset_descriptor, canonical_hash, regenerate

FAMILIES = ("wave", "burgers", "ks", "advection_diffusion", "allen_cahn",
            "gray_scott", "cahn_hilliard", "fitzhugh_nagumo")


def package_root():
    if os.environ.get("PHYSICS_BASELINE_PACKAGE"):
        return Path(os.environ["PHYSICS_BASELINE_PACKAGE"])
    for p in (Path(__file__).resolve().parents[1],
              Path.cwd() / "output" / "physics_calibrated_baseline"):
        if (p / "assets").is_dir():
            return p
    raise FileNotFoundError("Package assets not found; set PHYSICS_BASELINE_PACKAGE")


def development_source():
    if os.environ.get("PHYSICS_DEVELOPMENT_SOURCE"):
        return Path(os.environ["PHYSICS_DEVELOPMENT_SOURCE"])
    bundled = package_root() / "assets" / "development_physics.py"
    if bundled.is_file():
        return bundled
    return (Path.cwd() / "output" / "github_release" / "physics-guided-learning" /
            "experiments" / "development" / "lpb" / "physics.py")


class NumericalEquivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.torch_backend = create_backend("torch")
        cls.numpy_backend = create_backend("numpy")
        cls.torch_backend._torch.set_num_threads(1)

    def test_numpy_float64_pde_transitions_match_archived_torch(self):
        """Catch a wrong equation, mask, forcing, ETD coefficient, or substep."""
        tb, nb = self.torch_backend, self.numpy_backend
        maximum_error = 0.0
        cases = 0
        for grid in (32, 64):
            for family in FAMILIES:
                rng = np.random.default_rng(742001)
                params = np.concatenate((tb.physics.sample_params(family, 2, rng),
                                         tb.physics.sample_params(family, 2, rng, ood=True)))
                initial = tb.physics.initial_states(family, 4, grid, rng)
                actions = tb.physics.make_actions(family, 4, 4, rng)
                for prior in ("correct", "coefficient", "structural"):
                    p = tb.approx_params(family, params, prior)
                    for fidelity in ("coarse", "fine"):
                        with self.subTest(family=family, grid=grid, prior=prior, fidelity=fidelity):
                            ty, ny = initial.copy(), initial.copy()
                            for t in range(actions.shape[1]):
                                ty = tb.step(family, ty, actions[:, t], p, prior, fidelity, dtype=np.float64)
                                ny = nb.step(family, ny, actions[:, t], p, prior, fidelity, dtype=np.float64)
                                maximum_error = max(maximum_error, float(np.max(np.abs(ty - ny))))
                                np.testing.assert_allclose(ny, ty, rtol=5e-12, atol=5e-13)
                            cases += 1
        print(f"NumPy/PyTorch: {cases} four-step cases; max absolute difference {maximum_error:.3g}")

    def test_parameter_adjustment_preserves_archived_float32_rounding(self):
        """Avoid a silent extra-precision factor or double application of bias."""
        tb, nb = self.torch_backend, self.numpy_backend
        rng = np.random.default_rng(117)
        for dtype in (np.float32, np.float64):
            p = tb.physics.sample_params("wave", 12, rng).astype(dtype)
            for prior in ("correct", "coefficient", "structural"):
                expected = tb.approx_params("wave", p, prior)
                actual = nb.approx_params("wave", p, prior)
                self.assertEqual(actual.dtype, np.dtype(dtype))
                np.testing.assert_array_equal(actual, expected)

    def test_development_and_confirmation_existing_equations_match(self):
        """The newer module must preserve the six original PDE baselines."""
        source = development_source()
        if not source.is_file():
            self.skipTest("Historical development physics.py absent; set PHYSICS_DEVELOPMENT_SOURCE")
        module_spec = importlib.util.spec_from_file_location("development_physics_under_test", source)
        development = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(development)
        tb = self.torch_backend
        torch = tb._torch
        count = 0
        for family in FAMILIES[:6]:
            for grid in (32, 64):
                rng = np.random.default_rng(681113)
                p = development.sample_params(family, 4, rng, ood=True)
                y = development.initial_states(family, 4, grid, rng)
                a = development.make_actions(family, 4, 2, rng)
                for dtype in (np.float32, np.float64):
                    td = torch.float32 if dtype == np.float32 else torch.float64
                    for prior in ("correct", "coefficient", "structural"):
                        adjusted = tb.approx_params(family, p.astype(dtype), prior)
                        for fidelity in ("coarse", "fine"):
                            with self.subTest(family=family, grid=grid, dtype=dtype.__name__, prior=prior, fidelity=fidelity):
                                current = torch.tensor(y, dtype=td)
                                new = y.astype(dtype)
                                for t in range(2):
                                    with torch.no_grad():
                                        current = development.step(family, current, torch.tensor(a[:, t], dtype=td),
                                                                   torch.tensor(adjusted, dtype=td), prior, fidelity)
                                    new = tb.step(family, new, a[:, t], adjusted, prior, fidelity, dtype=dtype)
                                np.testing.assert_array_equal(current.numpy(), new)
                                count += 1
        print(f"Development/confirmation: {count} two-step cases bitwise equal")

    def test_full_wave_and_allen_cahn_datasets_match_archived_npz_hashes(self):
        """Reconstruct full historical arrays with exactly the original key order."""
        # The archive records 8 CPU threads. MKL FFT plans at 1 thread can
        # change roundoff before the final float32 cast, changing NPZ hashes.
        torch = self.torch_backend._torch
        previous_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous_threads)
        torch.set_num_threads(8)
        assets = package_root() / "assets" / "resolution_g64"
        cfg = json.loads((assets / "original_protocol.json").read_text())["config"]
        count = 0
        for family in ("wave", "allen_cahn"):
            for phase in ("development", "test"):
                with self.subTest(family=family, phase=phase):
                    archived = json.loads((assets / "data" / f"{family}_{phase}.json").read_text())
                    descriptor = dataset_descriptor(family, cfg, phase,
                                                     source_hash=archived["description"]["source_hash"])
                    self.assertEqual(descriptor, archived["description"])
                    self.assertEqual(canonical_hash(descriptor), archived["signature"])
                    data = regenerate(self.torch_backend, family, cfg, phase, smoke=False)
                    names = ("train", "val") if phase == "development" else ("test", "ood")
                    arrays = {}
                    for split in names:
                        for key in ("states", "actions", "params", "ids"):
                            arrays[f"{split}_{key}"] = data[split][key]
                    out = io.BytesIO()
                    np.savez_compressed(out, **arrays)
                    actual = hashlib.sha256(out.getvalue()).hexdigest()
                    self.assertEqual(actual, archived["sha256"],
                                     f"Archive NPZ mismatch for {family}/{phase}: actual {actual}; "
                                     "do not silently certify archive reproduction")
                    print(f"Exact archived NPZ: {family}/{phase} {actual}")
                    count += 1
        self.assertEqual(count, 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
