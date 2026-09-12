"""Matched forward-step timing without loading or training neural checkpoints.

The runner supplies the archived PyTorch step and models module. Importing this
module does not require PyTorch; the optional benchmark imports it on demand.
Fresh neural measurements describe initialized architecture execution only.
"""
from __future__ import annotations

import platform
import random
import statistics
import time


def benchmark_setting(step_torch, models_module, family, prior, cfg,
                      validation, normalization, multipliers, device="cpu",
                      blocks=5, repeats=20, batch_sizes=(1,), threads=1):
    """Return JSON-safe, interleaved timings for five predictor implementations.

    ``validation`` maps states/actions/supplied_params to arrays with shapes
    [trajectory,time,channel,grid], [trajectory,time,1], and [trajectory,3].
    Parameters are already biased when appropriate; this function never calls
    approx_params. ``multipliers`` is the three fitted positive corrections.

    Only the first observed context from validation is used. Accuracy is not
    scored. All methods share resident float32 inputs on the requested device;
    original internal solver arithmetic is preserved. Physical methods use the
    last observed state/action. Neural methods also consume observed history.
    The calibrated solver includes multiplication by the fitted correction.
    """
    import numpy as np
    import torch

    started = time.perf_counter()
    blocks, repeats, threads = int(blocks), int(repeats), int(threads)
    batch_sizes = tuple(int(value) for value in batch_sizes)
    if not batch_sizes or min(blocks, repeats, threads, *batch_sizes) < 1:
        raise ValueError("blocks, repeats, threads and batch sizes must be positive")
    if len(set(batch_sizes)) != len(batch_sizes):
        raise ValueError("batch sizes must be distinct")
    if prior not in ("correct", "coefficient", "structural"):
        raise ValueError(f"Unsupported prior {prior!r}")
    target = torch.device(device)
    if target.type not in ("cpu", "cuda"):
        raise ValueError("Matched timing supports CPU or CUDA devices")
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; explicitly choose CPU timing or a working GPU environment")

    context = int(cfg["context"])
    states = np.asarray(validation["states"])
    actions = np.asarray(validation["actions"])
    supplied = np.asarray(validation["supplied_params"])
    correction = np.asarray(multipliers, dtype=np.float64)
    if states.ndim != 4 or actions.ndim != 3 or supplied.ndim != 2:
        raise ValueError("Expected states [B,T,C,G], actions [B,T,1], supplied_params [B,3]")
    count, steps, channels, grid = states.shape
    if (context < 1 or steps < context or actions.shape[1] < context or
            actions.shape[0] != count or actions.shape[2] != 1 or
            supplied.shape != (count, 3) or max(batch_sizes) > count):
        raise ValueError("Validation shapes, available trajectories and context are inconsistent")
    if int(cfg.get("grid", grid)) != grid:
        raise ValueError("Configuration grid differs from the validation grid")
    if correction.shape != (3,) or not np.isfinite(correction).all() or np.any(correction <= 0):
        raise ValueError("multipliers must contain three finite positive fitted corrections")
    for key, size in (("mean", channels), ("scale", channels),
                      ("param_mean", 3), ("param_scale", 3)):
        values = np.asarray(normalization[key], dtype=np.float64)
        if values.shape != (size,) or not np.isfinite(values).all():
            raise ValueError(f"Invalid normalization field {key!r}")
        if key.endswith("scale") and np.any(values <= 0):
            raise ValueError(f"Normalization field {key!r} must be positive")

    class RawPredictor(torch.nn.Module):
        """Exact normalization/denormalization used by archived training.py."""
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.register_buffer("state_mean", torch.tensor(normalization["mean"], dtype=torch.float32)[None, None, :, None])
            self.register_buffer("state_scale", torch.tensor(normalization["scale"], dtype=torch.float32)[None, None, :, None])
            self.register_buffer("param_mean", torch.tensor(normalization["param_mean"], dtype=torch.float32)[None, :])
            self.register_buffer("param_scale", torch.tensor(normalization["param_scale"], dtype=torch.float32)[None, :])

        def forward(self, history, action_history, params):
            normalized = self.model(
                (history - self.state_mean) / self.state_scale, action_history,
                (params - self.param_mean) / self.param_scale)
            return normalized * self.state_scale[:, 0] + self.state_mean[:, 0]

    def sync():
        if target.type == "cuda":
            torch.cuda.synchronize(target)

    previous_threads = torch.get_num_threads()
    rows, block_orders = [], []
    order_rng = random.Random(20260913)
    try:
        torch.set_num_threads(threads)
        keep = max(batch_sizes)
        history = torch.as_tensor(np.ascontiguousarray(states[:keep, :context]), dtype=torch.float32, device=target)
        action_history = torch.as_tensor(np.ascontiguousarray(actions[:keep, :context]), dtype=torch.float32, device=target)
        params = torch.as_tensor(np.ascontiguousarray(supplied[:keep]), dtype=torch.float32, device=target)
        fitted = torch.as_tensor(correction, dtype=torch.float32, device=target)[None, :]
        if not all(bool(torch.isfinite(value).all().item()) for value in (history, action_history, params, fitted)):
            raise ValueError("Timing inputs must remain finite when represented in float32")
        # Initialize on CPU and restore its RNG so this optional benchmark does
        # not change random draws in the parent experiment. No optimizer exists.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(20260913)
            predictors = {
                kind: RawPredictor(models_module.build_model(
                    kind, {"channels": channels, "grid": grid}, cfg)).to(target).eval()
                for kind in ("transformer", "looped", "fno")
            }
        neural_counts = {
            kind: sum(parameter.numel() * (2 if parameter.is_complex() else 1)
                      for parameter in model.parameters())
            for kind, model in predictors.items()
        }

        with torch.no_grad():
            for batch_size in batch_sizes:
                h, a, p = history[:batch_size], action_history[:batch_size], params[:batch_size]
                # Input views and history bookkeeping are prepared outside the
                # measured call, identically for the two physical predictors.
                last_state, last_action = h[:, -1], a[:, -1]
                calls = {
                    "physics_uncalibrated": lambda: step_torch(
                        family, last_state, last_action, p, mismatch=prior, fidelity="coarse"),
                    "physics_calibrated": lambda: step_torch(
                        family, last_state, last_action, p * fitted, mismatch=prior, fidelity="coarse"),
                }
                for kind, model in predictors.items():
                    calls[kind] = lambda model=model: model(h, a, p)
                measurements = {name: [] for name in calls}
                for name, call in calls.items():
                    for _ in range(5):
                        output = call()
                    # Check once outside timed regions; do not certify accuracy.
                    if output.shape != last_state.shape or not bool(torch.isfinite(output).all().item()):
                        raise FloatingPointError(f"{family}/{prior}/{name}: invalid warmup output")
                sync()
                for block in range(blocks):
                    order = list(calls)
                    order_rng.shuffle(order)
                    block_orders.append({"batch_size": batch_size, "block": block, "order": order})
                    for name in order:
                        call = calls[name]
                        sync()
                        start_ns = time.perf_counter_ns()
                        for _ in range(repeats):
                            call()
                        sync()
                        elapsed_ns = time.perf_counter_ns() - start_ns
                        measurements[name].append(elapsed_ns / (1e6 * repeats))
                for kind, values in measurements.items():
                    is_neural = kind in predictors
                    rows.append({
                        "family": family, "prior": prior, "grid": grid,
                        "kind": kind, "batch_size": batch_size,
                        "median_ms_per_batch_step": statistics.median(values),
                        "min_ms_per_batch_step": min(values),
                        "max_ms_per_batch_step": max(values),
                        "block_ms_per_batch_step": values,
                        "parameter_count": neural_counts.get(kind),
                        "parameter_count_convention": "real scalar equivalents; complex neural weights count twice" if is_neural else None,
                        "fitted_global_multiplier_count": 3 if kind == "physics_calibrated" else 0,
                        "weights": "initialized; no checkpoint; no training" if is_neural else "physical coefficients supplied per trajectory",
                        "timing_source": "fresh_matched_execution_benchmark",
                        "calibration_multiplication_included": kind == "physics_calibrated",
                    })
        environment = {
            "python": platform.python_version(), "platform": platform.platform(),
            "processor": platform.processor(), "numpy": np.__version__,
            "torch": str(torch.__version__), "device": str(target),
            "gpu": torch.cuda.get_device_name(target) if target.type == "cuda" else None,
            "cuda_runtime": torch.version.cuda,
            "threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        }
    finally:
        torch.set_num_threads(previous_threads)

    return {
        "status": "COMPLETED_EXECUTION_BENCHMARK",
        "neural_training": False, "neural_checkpoints_loaded": False,
        "accuracy_evaluation": False,
        "family": family, "prior": prior, "grid": grid,
        "multipliers": correction.tolist(), "context": context,
        "blocks": blocks, "repeats_per_block": repeats, "warmup_calls_per_method": 5,
        "batch_sizes": list(batch_sizes), "block_orders": block_orders,
        "elapsed_seconds": time.perf_counter() - started,
        "environment": environment, "rows": rows,
        "timing_scope": (
            "Resident float32 validation inputs on one device. One batch-step forward call; "
            "full neural normalization/denormalization or the archived coarse physical step. "
            "Calibrated physics includes applying fitted multipliers. Transfer, input-view "
            "creation, history updates, fitting, and forecast-error calculation are excluded. "
            "Physical predictors consume the last state/action; neural predictors consume "
            "the observed context. CUDA synchronization brackets each repeated block."
        ),
        "limitations": [
            "Neural weights are initialized solely for execution timing. Historical accuracy uses archived trained models and remains a separate measurement.",
            "Forward graphs have fixed dimensions and zero-initialized output heads; all layers execute. No trained-checkpoint latency is claimed.",
            "Float32 refers to resident inputs. The archived solver retains float64/complex128 exponential-coefficient calculations and its existing transforms and substeps.",
            "This times the existing solver implementation without coefficient caching, solver optimization, compilation, or a tolerance sweep.",
            "Fixed validation contexts are reused. Measurements describe one-step execution, not complete rollout wall time or sustained deployment throughput.",
            "Structural omissions remain in both physical implementations; calibration only multiplies supplied coefficients.",
            "Fresh physical and neural latency rows are comparable within this call. They must not be pooled with historical timings from other hardware.",
        ],
    }
