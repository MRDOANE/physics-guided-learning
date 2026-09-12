"""Fixed-budget training, exact-state resume, and paired rollout evaluation.

The physics intervention is a detached coarse discrete-transition consistency
penalty on additional unlabeled, perturbed training contexts. It is not a differential-equation PINN residual. No FLOP or parameter
matching is asserted. Normalization uses only the selected training prefix.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn

from .models import build_model
from . import physics


SCHEMA_VERSION = 1


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temp, path)


def _save_checkpoint(path, payload):
    """Separate helper makes interrupt/resume testable at an actual commit boundary."""
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def _load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # Compatibility with older user-provided PyTorch installations.
        return torch.load(path, map_location=device)


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _fingerprint(data):
    digest = hashlib.sha256()
    digest.update(json.dumps(data["spec"], sort_keys=True).encode())
    for split in ("train", "val", "test", "ood"):
        for key in ("states", "actions", "params"):
            arr = np.ascontiguousarray(data[split][key])
            digest.update(f"{split}:{key}:{arr.shape}:{arr.dtype}".encode())
            digest.update(memoryview(arr).cast("B"))
    return digest.hexdigest()


def _seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # On a fixed software/hardware stack this improves reproducibility. Cross-device
    # bitwise equality is not promised, and unsupported deterministic ops raise.
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False


def _rng_state(sampler):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "sampler": copy.deepcopy(sampler.bit_generator.state)}


def _restore_rng(state, sampler):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume needs the same visible CUDA device count")
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])
    sampler.bit_generator.state = state["sampler"]


def _prepare(system, data, cfg):
    n = int(cfg["data_count"])
    if n < 1 or n > len(data["train"]["states"]):
        raise ValueError("data_count is outside available training trajectories")
    selected = np.asarray(data["train"]["states"][:n], dtype=np.float64)
    mean = selected.mean(axis=(0, 1, 3), keepdims=False).astype(np.float32)
    scale = np.maximum(selected.std(axis=(0, 1, 3)), 1e-6).astype(np.float32)
    tensors = {}
    for split in (name for name in ("train", "val", "test", "ood") if name in data):
        src = data[split]
        stop = n if split == "train" else len(src["states"])
        tensors[split] = {k: torch.as_tensor(np.array(src[k][:stop], copy=True), dtype=torch.float32)
                          for k in ("states", "actions", "params")}
        if stop < 1:
            raise ValueError(f"Empty {split} split")
        tensors[split]["params"] = physics.approx_params(system, tensors[split]["params"], cfg["mismatch"])
        if tensors[split]["states"].shape[1] < int(cfg["context"]) + 1:
            raise ValueError(f"{split} trajectories shorter than context + one target")
    pc = tensors["train"]["params"].mean(dim=0)
    ps = tensors["train"]["params"].std(dim=0, unbiased=False)
    ps = torch.maximum(ps, torch.maximum(pc.abs() * 0.05, torch.full_like(ps, 1e-4)))
    norm = {"mean": mean.tolist(), "scale": scale.tolist(),
            "param_mean": pc.tolist(), "param_scale": ps.tolist()}
    return tensors, norm


class RawPredictor(nn.Module):
    """Adapter around a normalized network; coefficients must already be approximate."""
    def __init__(self, model, normalization):
        super().__init__()
        self.model = model
        self.register_buffer("state_mean", torch.tensor(normalization["mean"], dtype=torch.float32)[None, None, :, None])
        self.register_buffer("state_scale", torch.tensor(normalization["scale"], dtype=torch.float32)[None, None, :, None])
        self.register_buffer("param_mean", torch.tensor(normalization["param_mean"], dtype=torch.float32)[None, :])
        self.register_buffer("param_scale", torch.tensor(normalization["param_scale"], dtype=torch.float32)[None, :])

    def normalized(self, history, actions, params):
        return self.model((history - self.state_mean) / self.state_scale, actions,
                          (params - self.param_mean) / self.param_scale)

    def forward(self, history, actions, params):
        z = self.normalized(history, actions, params)
        return z * self.state_scale[:, 0] + self.state_mean[:, 0]


def load_predictor(run_dir, device="cpu"):
    checkpoint = _load_checkpoint(Path(run_dir) / "checkpoint.pt", device)
    model = build_model(checkpoint["config"]["kind"], checkpoint["spec"], checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["best_model"])
    predictor = RawPredictor(model, checkpoint["normalization"]).to(device).eval()
    metadata = {k: checkpoint[k] for k in ("config", "system", "spec", "normalization", "best_step")}
    return predictor, metadata


def _batch(split, ids, times, context, device):
    ti = torch.as_tensor(ids, dtype=torch.long)
    ts = torch.as_tensor(times, dtype=torch.long)
    offsets = torch.arange(context, dtype=torch.long)
    windows = ts[:, None] - (context - 1) + offsets[None, :]
    history = split["states"][ti[:, None], windows].to(device)
    actions = split["actions"][ti[:, None], windows].to(device)
    params = split["params"][ti].to(device)
    target = split["states"][ti, ts + 1].to(device)
    return history, actions, params, target


def _validation(predictor, split, cfg, device):
    predictor.eval()
    k = int(cfg["context"])
    count, transitions = len(split["states"]), split["actions"].shape[1] - k + 1
    total = count * transitions
    indices = np.linspace(0, total - 1, min(total, int(cfg.get("val_samples", 512))), dtype=np.int64)
    squared, numel = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(indices), int(cfg["batch_size"])):
            chunk = indices[start:start + int(cfg["batch_size"])]
            hist, acts, params, target = _batch(split, chunk // transitions, chunk % transitions + k - 1, k, device)
            pred = predictor.normalized(hist, acts, params)
            true = (target - predictor.state_mean[:, 0]) / predictor.state_scale[:, 0]
            error = (pred - true).square()
            if not torch.isfinite(error).all():
                return float("inf")
            squared += float(error.sum().cpu())
            numel += error.numel()
    return squared / max(1, numel)


def _physics_collocation(system, split, predictor, cfg, device):
    """Cache deterministic perturbed contexts and coarse pseudo-labels once/run.

    A physics penalty on the clean noiseless supervised transitions would mostly
    duplicate their labels. Here each observed training context supplies a nearby
    unlabeled context: independent Gaussian state perturbations, with channelwise
    standard deviation physics_jitter * training scale. No oracle is consulted.
    The same permitted actions and approximate coefficients remain unchanged.
    """
    k = int(cfg["context"])
    n = len(split["states"])
    t = split["actions"].shape[1] - k + 1
    c, g = split["states"].shape[2:]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(cfg["seed"]) + 3907)
    sigma = predictor.state_scale.detach().cpu() * float(cfg["physics_jitter"])
    histories, targets = [], []
    with torch.no_grad():
        for start in range(0, n * t, 256):
            flat = np.arange(start, min(start + 256, n * t))
            hist, acts, params, _ = _batch(split, flat // t, flat % t + k - 1, k, "cpu")
            noise = torch.randn(hist.shape, generator=generator, dtype=hist.dtype)
            perturbed = hist + sigma * noise
            z = physics.step(system, perturbed[:, -1].to(device), acts[:, -1].to(device),
                             params.to(device), mismatch=cfg["mismatch"], fidelity="coarse")
            if not torch.isfinite(z).all():
                raise FloatingPointError("Nonfinite collocation target: coarse solver must be audited before training")
            z = (z - predictor.state_mean[:, 0]) / predictor.state_scale[:, 0]
            histories.append(perturbed)
            targets.append(z.cpu())
    return (torch.cat(histories).reshape(n, t, k, c, g),
            torch.cat(targets).reshape(n, t, c, g))


def _rollout(predictor, system, split, cfg, device, method):
    k = int(cfg["context"])
    horizon = min(int(cfg["eval_horizon"]), split["actions"].shape[1] - k + 1)
    if horizon < 1:
        raise ValueError("eval_horizon must be positive")
    all_errors, all_failed = [], []
    predictor.eval()
    threshold = float(cfg.get("divergence_threshold", 1e6))
    with torch.no_grad():
        for start in range(0, len(split["states"]), int(cfg["batch_size"])):
            states = split["states"][start:start + int(cfg["batch_size"])].to(device)
            actions = split["actions"][start:start + int(cfg["batch_size"])].to(device)
            params = split["params"][start:start + int(cfg["batch_size"])] .to(device)
            history = states[:, :k].clone()
            failed = torch.zeros(len(states), dtype=torch.bool, device=device)
            errors = []
            for h in range(horizon):
                acts = actions[:, h:h + k]
                if method == "model":
                    prediction = predictor(history, acts, params)
                elif method == "mechanistic":
                    prediction = physics.step(system, history[:, -1], acts[:, -1], params,
                                              mismatch=cfg["mismatch"], fidelity="coarse")
                elif method == "persistence":
                    prediction = history[:, -1]
                else:
                    raise ValueError(method)
                normalized = (prediction - predictor.state_mean[:, 0]) / predictor.state_scale[:, 0]
                bad = ~torch.isfinite(normalized).flatten(1).all(dim=1)
                bad |= normalized.abs().flatten(1).amax(dim=1) > threshold
                failed |= bad
                error = ((prediction - states[:, k + h]) / predictor.state_scale[:, 0]).square().flatten(1).mean(dim=1)
                error[failed] = float("inf")
                errors.append(error.cpu())
                # Failed trajectories remain scored as infinite. Their computational
                # placeholders are reset solely to keep batched arithmetic finite.
                prediction = torch.where(failed[:, None, None], predictor.state_mean[:, 0].expand_as(prediction), prediction)
                history = torch.cat((history[:, 1:], prediction[:, None]), dim=1)
            all_errors.append(torch.stack(errors, dim=1).numpy())
            all_failed.append(failed.cpu().numpy())
    curve = np.concatenate(all_errors, axis=0).astype(np.float64)
    failed = np.concatenate(all_failed)
    per_trajectory = curve.mean(axis=1)
    metrics = {"normalized_rmse": float(np.sqrt(per_trajectory.mean())),
               "final_normalized_rmse": float(np.sqrt(curve[:, -1].mean())),
               "divergence_fraction": float(failed.mean()), "n_trajectories": len(curve),
               "horizon": horizon}
    return metrics, per_trajectory, curve, failed


def _latency(predictor, split, cfg, device):
    # Batch-one latency, on the selected device, including normalization and the
    # complete depth of the predictor. Warmups are excluded; CUDA is synchronized.
    k = int(cfg["context"])
    hist, acts, params, _ = _batch(split, [0], [k - 1], k, device)
    predictor.eval()
    with torch.no_grad():
        for _ in range(5):
            predictor(hist, acts, params)
        _sync(device)
        start = time.perf_counter()
        for _ in range(20):
            predictor(hist, acts, params)
        _sync(device)
    return 1000.0 * (time.perf_counter() - start) / 20

