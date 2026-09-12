"""Archive-faithful dataset regeneration and explicit numerical backends.

The torch backend delegates to the unmodified archived_lpb.physics module. The
NumPy backend is a float64 diagnostic port. Its float32 output mode rounds the
float64 result; it does not reproduce every float32 PyTorch intermediate.

Test generation is a low-level primitive. The runner must freeze calibration
and selection decisions before calling regenerate(..., phase='test').
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib
import json
from typing import Any

import numpy as np

_SPLITS = ("train", "val", "test", "ood")


def canonical_hash(value: Any) -> str:
    """Identical JSON hash convention to archived_lpb.core.canonical_hash."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _dtype(dtype) -> np.dtype:
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("Only float32 and float64 backend step dtypes are supported")
    return dtype


class _NumpyBackend:
    name = "numpy"
    diagnostic_only = True
    exact_archived_arithmetic = False
    float32_semantics = "Float64 NumPy step, then cast to float32; diagnostic only."

    def __init__(self):
        package_prefix = f"{__package__}." if __package__ else ""
        self.physics = importlib.import_module(package_prefix + "numpy_physics")

    def step(self, name, state, action, params, mismatch="correct", fidelity="coarse",
             dtype=np.float64):
        dtype = _dtype(dtype)
        # Rounding inputs makes float32 input representation explicit. Numerical
        # operations in numpy_physics remain float64 by design.
        state = np.asarray(state, dtype=dtype)
        action = np.asarray(action, dtype=dtype)
        params = np.asarray(params, dtype=dtype)
        result = self.physics.step(name, state, action, params, mismatch=mismatch,
                                   fidelity=fidelity)
        return np.asarray(result, dtype=dtype)

    def approx_params(self, family, params, prior):
        self.physics.spec(family)
        params = np.asarray(params)
        dtype = params.dtype if params.dtype in (np.dtype(np.float32), np.dtype(np.float64)) else np.dtype(np.float64)
        params = np.asarray(params, dtype=dtype)
        if prior in ("correct", "structural"):
            return params.copy()
        if prior == "coefficient":
            return params * np.asarray([0.70, 1.30, 1.40], dtype=dtype)
        raise ValueError(f"Unknown mismatch {prior!r}")

    def simulate(self, name, params, initial, actions, fidelity="fine"):
        return self.physics.simulate(name, params, initial, actions, fidelity=fidelity)


class _TorchBackend:
    name = "torch"
    diagnostic_only = False
    exact_archived_arithmetic = True
    float32_semantics = "Unmodified archived PyTorch float32 step on CPU."

    def __init__(self):
        # Import torch only on explicit selection. Importing this module or using
        # NumPy does not require PyTorch to be installed.
        try:
            self._torch = importlib.import_module("torch")
        except ImportError as exc:
            raise RuntimeError("The torch backend requires PyTorch. Use the supplied CPU "
                               "environment or select numpy for diagnostics only.") from exc
        try:
            self.physics = importlib.import_module("archived_lpb.physics")
        except ImportError as exc:
            raise RuntimeError("The unmodified archived_lpb/physics.py must be bundled "
                               "beside the runner for the torch backend.") from exc

    def step(self, name, state, action, params, mismatch="correct", fidelity="coarse",
             dtype=np.float64):
        dtype = _dtype(dtype)
        torch = self._torch
        td = torch.float64 if dtype == np.dtype(np.float64) else torch.float32
        state = torch.as_tensor(np.ascontiguousarray(state, dtype=dtype), dtype=td, device="cpu")
        action = torch.as_tensor(np.ascontiguousarray(action, dtype=dtype), dtype=td, device="cpu")
        params = torch.as_tensor(np.ascontiguousarray(params, dtype=dtype), dtype=td, device="cpu")
        with torch.no_grad():
            result = self.physics.step(name, state, action, params, mismatch=mismatch,
                                       fidelity=fidelity)
        return result.detach().cpu().numpy().copy()

    def approx_params(self, family, params, prior):
        params = np.asarray(params)
        dtype = params.dtype if params.dtype in (np.dtype(np.float32), np.dtype(np.float64)) else np.dtype(np.float64)
        torch = self._torch
        td = torch.float32 if dtype == np.dtype(np.float32) else torch.float64
        tensor = torch.as_tensor(np.ascontiguousarray(params, dtype=dtype), dtype=td, device="cpu")
        with torch.no_grad():
            result = self.physics.approx_params(family, tensor, prior)
        return result.detach().cpu().numpy().copy()

    def simulate(self, name, params, initial, actions, fidelity="fine"):
        return self.physics.simulate(name, params, initial, actions, fidelity=fidelity)


def create_backend(name: str):
    """Return an explicit CPU backend; no CUDA initialization is performed."""
    if name == "numpy":
        return _NumpyBackend()
    if name == "torch":
        return _TorchBackend()
    raise ValueError(f"Unknown backend {name!r}; choose 'torch' or 'numpy'")


def effective_config(cfg: dict[str, Any], smoke: bool = False) -> dict[str, Any]:
    """Copy caller configuration and apply only the declared smoke reductions."""
    out = deepcopy(cfg)
    if smoke:
        out["split_counts"] = {"train": 8, "val": 4, "test": 8, "ood": 8}
        out["trajectory_steps"] = max(64, int(out.get("context", 4)) + 32)
        out["burn_in"] = 24
    required = ("grid", "trajectory_steps", "burn_in", "split_counts", "train_val_seed", "test_seed")
    missing = [key for key in required if key not in out]
    if missing:
        raise ValueError(f"Missing archive dataset configuration fields: {missing}")
    if int(out["train_val_seed"]) == int(out["test_seed"]):
        raise ValueError("Development and test generation seeds must be distinct")
    for key in ("grid", "trajectory_steps"):
        if int(out[key]) < 1:
            raise ValueError(f"{key} must be positive")
    if int(out["burn_in"]) < 0:
        raise ValueError("burn_in must be nonnegative")
    for split in _SPLITS:
        if split not in out["split_counts"] or int(out["split_counts"][split]) < 1:
            raise ValueError(f"split_counts[{split!r}] must be positive")
    return out


def dataset_descriptor(family: str, cfg: dict[str, Any], phase: str,
                       smoke: bool = False, source_hash: str | None = None) -> dict[str, Any]:
    """Reconstruct archive descriptor fields.

    Supply the archived source hash to reproduce its full signature. With no
    source hash supplied, that field is None and the descriptor is informative;
    its signature must not be claimed identical to the archived signature.
    """
    if phase not in ("development", "test"):
        raise ValueError(f"Unknown generation phase {phase!r}")
    cfg = effective_config(cfg, smoke)
    names = ("train", "val") if phase == "development" else ("test", "ood")
    if source_hash is None:
        source_hash = cfg.get("source_hash")
    return dict(family=family, phase=phase, source_hash=source_hash,
                grid=cfg["grid"], steps=cfg["trajectory_steps"], burn_in=cfg["burn_in"],
                counts={name: cfg["split_counts"][name] for name in names},
                seed=cfg["train_val_seed"] if phase == "development" else cfg["test_seed"])


def regenerate(backend, family: str, cfg: dict[str, Any], phase: str,
               smoke: bool = False) -> dict[str, dict[str, np.ndarray]]:
    """Regenerate whole-trajectory splits with the archived RNG and arithmetic.

    Output float32 arrays match the archived data representation. The torch
    backend uses the original CPU float64 generator before the final casts.
    NumPy is diagnostic until numerical agreement has been independently checked.
    No trajectory is discarded, replaced, clipped, or reseeded on a failure.
    """
    cfg = effective_config(cfg, smoke)
    descriptor = dataset_descriptor(family, cfg, phase)
    physics = backend.physics
    physics.spec(family, cfg["grid"])
    family_seed = int(canonical_hash(family)[:8], 16)
    result = {}
    for split in descriptor["counts"]:
        split_index = _SPLITS.index(split)
        rng = np.random.default_rng(np.random.SeedSequence(
            [descriptor["seed"], family_seed, split_index]))
        n = cfg["split_counts"][split]
        burn = cfg["burn_in"]
        # Sampling order and batch sizes are part of archive reproducibility.
        params = physics.sample_params(family, n, rng, ood=(split == "ood"))
        initial = physics.initial_states(family, n, cfg["grid"], rng)
        actions = physics.make_actions(family, n, cfg["trajectory_steps"] + burn, rng)
        states = np.concatenate([
            backend.simulate(family, params[i:i + 32], initial[i:i + 32],
                             actions[i:i + 32], fidelity="fine")
            for i in range(0, n, 32)
        ])[:, burn:]
        if not np.isfinite(states).all():
            raise FloatingPointError(f"Nonfinite numerical reference {family}/{split}; "
                                     "no trajectories discarded")
        ids = np.array([f'{family}:{descriptor["seed"]}:{split}:{i}' for i in range(n)])
        result[split] = {
            "states": states.astype(np.float32),
            "actions": actions[:, burn:].astype(np.float32),
            "params": params.astype(np.float32),
            "trajectory_ids": ids,
            "ids": ids,  # Archive key spelling, for compatibility.
        }
    return result


def array_sha256(array: np.ndarray) -> str:
    """Content hash independent of NPZ compression and ZIP metadata."""
    array = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(canonical_hash({"dtype": array.dtype.str, "shape": list(array.shape)}).encode())
    h.update(array.tobytes(order="C"))
    return h.hexdigest()


def split_array_hashes(data: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, str]]:
    return {split: {key: array_sha256(arrays[key]) for key in ("states", "actions", "params", "ids")}
            for split, arrays in data.items()}
