"""Differentiable, dimensionless controlled dynamical systems.

All spectral systems use a periodic, equispaced grid without a duplicated endpoint.
See docs/PHYSICS.md for equations, discretization and the limits of the reference.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

_SYSTEMS = {
    "reactor": dict(channels=2, grid=1, dt=0.08, length=None,
                    param_names=["damkohler", "activation", "cooling"],
                    param_center=[0.35, 2.0, 0.8]),
    "pendulum": dict(channels=2, grid=1, dt=0.08, length=None,
                     param_names=["gravity_frequency_sq", "damping", "inverse_inertia"],
                     param_center=[4.0, 0.15, 1.0]),
    "burgers": dict(channels=1, grid=32, dt=0.1, length=2 * math.pi,
                    param_names=["viscosity", "advection", "drag"],
                    param_center=[0.08, 1.0, 0.05]),
    "wave": dict(channels=2, grid=32, dt=0.05, length=2 * math.pi,
                 param_names=["wave_speed", "damping", "foundation_stiffness"],
                 param_center=[1.0, 0.1, 0.2]),
    "wave_high": dict(channels=2, grid=32, dt=0.05, length=2 * math.pi,
                 param_names=["wave_speed", "damping", "foundation_stiffness"],
                 param_center=[1.0, 0.1, 0.2], forcing_mode=4),
    "ks": dict(channels=1, grid=32, dt=0.25, length=32.0,
               param_names=["destabilizing_diffusion", "hyperdiffusion", "advection"],
               param_center=[1.0, 1.0, 1.0]),
    "advection_diffusion": dict(channels=1, grid=32, dt=0.1, length=2 * math.pi,
               param_names=["diffusivity", "transport_speed", "drag"],
               param_center=[0.06, 1.0, 0.04]),
    "allen_cahn": dict(channels=1, grid=32, dt=0.1, length=2 * math.pi,
               param_names=["diffusivity", "reaction_rate", "forcing_strength"],
               param_center=[0.035, 1.0, 0.2]),
    "gray_scott": dict(channels=2, grid=32, dt=0.5, length=32.0,
               param_names=["substrate_diffusivity", "feed_rate", "removal_rate"],
               param_center=[0.16, 0.03, 0.04]),
}
_FIDELITY = {"coarse": 1, "fine": 4, "check": 8}


def spec(name: str, grid: int = 32) -> dict[str, Any]:
    if name not in _SYSTEMS:
        raise ValueError(f"Unknown system {name!r}; choose {tuple(_SYSTEMS)}")
    out = {"name": name, **_SYSTEMS[name]}
    out["param_center"] = list(out["param_center"])
    out["param_names"] = list(out["param_names"])
    if out["grid"] > 1:
        minimum_grid = 32 if name == "ks" else 16
        if grid < minimum_grid or grid % 2:
            reason = "; KS needs resolved dissipative modes after 2/3 truncation" if name == "ks" else ""
            raise ValueError(f"{name} grid must be an even integer >= {minimum_grid}{reason}")
        out["grid"] = int(grid)
    return out


def sample_params(name: str, count: int, rng: np.random.Generator,
                  ood: bool = False) -> np.ndarray:
    """ID coefficients are 0.85--1.15 times center; OOD 1.30--1.50.

    All coordinates shift together, a declared joint coefficient extrapolation
    experiment, not comprehensive testing of every possible distribution shift.
    """
    center = np.asarray(spec(name)["param_center"], dtype=np.float64)
    low, high = (1.30, 1.50) if ood else (0.85, 1.15)
    return center[None] * rng.uniform(low, high, (count, 3))


def initial_states(name: str, count: int, grid: int,
                   rng: np.random.Generator) -> np.ndarray:
    s = spec(name, grid)
    if name == "reactor":
        return np.stack((rng.uniform(0.25, 1.0, count),
                         rng.uniform(0.05, 0.6, count)), axis=1)[..., None]
    if name == "pendulum":
        return np.stack((rng.uniform(-math.pi, math.pi, count),
                         rng.uniform(-0.6, 0.6, count)), axis=1)[..., None]
    phase = np.linspace(0, 2 * math.pi, s["grid"], endpoint=False)
    out = np.zeros((count, s["channels"], s["grid"]), dtype=np.float64)
    if name == "gray_scott":
        # Smooth finite-amplitude concentrations, with independently shifted
        # one- and two-lobed seeds. The initialization is positive without clips.
        phase1 = rng.uniform(0, 2 * math.pi, (count, 1))
        phase2 = rng.uniform(0, 2 * math.pi, (count, 1))
        bump = (0.6 * np.exp(3 * (np.cos(phase[None] - phase1) - 1)) +
                0.4 * np.exp(4 * (np.cos(2 * phase[None] - phase2) - 1)))
        out[:, 0] = 0.8 - 0.35 * bump
        out[:, 1] = 0.08 + rng.uniform(0.19, 0.25, (count, 1)) * bump
        return out
    modes = min(5 if name == "ks" else 3, s["grid"] // 3)
    amplitude = {"burgers": 0.6, "wave": 0.35, "wave_high": 0.35, "ks": 0.8,
                 "advection_diffusion": 0.6, "allen_cahn": 0.4}[name]
    for c in range(s["channels"]):
        for mode in range(1, modes + 1):
            a = rng.uniform(0.5, 1.0, (count, 1)) * amplitude / mode
            if c == 1:
                a *= 0.5
            p = rng.uniform(0, 2 * math.pi, (count, 1))
            spatial_mode = mode + 3 if name == "wave_high" else mode
            out[:, c] += a * np.sin(spatial_mode * phase[None] + p)
    if name == "burgers":
        out[:, 0] += rng.uniform(-0.2, 0.2, (count, 1))
    return out


def make_actions(name: str, count: int, steps: int,
                 rng: np.random.Generator) -> np.ndarray:
    """Known, bounded piecewise-linear drive, held constant within each step."""
    spec(name)
    knots = np.arange(0, steps + 8, 8)
    if knots[-1] < steps:
        knots = np.append(knots, steps)
    values = rng.uniform(-1, 1, (count, len(knots)))
    out = np.stack([np.interp(np.arange(steps), knots, v) for v in values])
    if name == "pendulum":
        out *= 2.0
    elif name == "reactor":
        out = 0.1 + 0.15 * out
    return out[..., None].astype(np.float64)


def approx_params(name: str, params: torch.Tensor, mismatch: str) -> torch.Tensor:
    spec(name)
    if mismatch in ("correct", "structural"):
        return params.clone()
    if mismatch == "coefficient":
        # Fixed multiplicative error shared across all systems: no true values
        # are subsequently supplied as additional model features.
        return params * params.new_tensor([0.70, 1.30, 1.40])
    raise ValueError(f"Unknown mismatch {mismatch!r}")


def _rk4(state: torch.Tensor, rhs, h: float) -> torch.Tensor:
    k1 = rhs(state)
    k2 = rhs(state + (h / 2) * k1)
    k3 = rhs(state + (h / 2) * k2)
    k4 = rhs(state + h * k3)
    return state + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def _wave_numbers(state: torch.Tensor, length: float) -> torch.Tensor:
    return 2 * math.pi * torch.fft.rfftfreq(
        state.shape[-1], d=length / state.shape[-1],
        device=state.device, dtype=state.dtype)


def _phi(z: torch.Tensor, order: int) -> torch.Tensor:
    """Stable real exponential phi functions, including exact zero eigenvalues."""
    small = z.abs() < 0.2
    safe = torch.where(small, torch.ones_like(z), z)
    em = torch.expm1(z)
    if order == 1:
        direct = em / safe
    elif order == 2:
        direct = (em - z) / safe.square()
    else:
        direct = (em - z - z.square() / 2) / safe.pow(3)
    # Degree 10 Taylor expansion has < 1e-18 absolute truncation error at 0.2.
    series = torch.full_like(z, 1.0 / math.factorial(order + 10))
    for j in range(9, -1, -1):
        series = series * z + 1.0 / math.factorial(order + j)
    return torch.where(small, series, direct)


def _spectral_step(name: str, state: torch.Tensor, action: torch.Tensor,
                   params: torch.Tensor, structural: bool, substeps: int,
                   dt: float, length: float) -> torch.Tensor:
    """Cox--Matthews ETDRK4 with a declared 2/3-truncated quadratic term."""
    n = state.shape[-1]
    k = _wave_numbers(state, length)[None, :]
    a, b, c = (params[:, i:i + 1] for i in range(3))
    if name == "burgers":
        linear = -a * k.square() - c
        nonlinear_coeff = b if not structural else torch.zeros_like(b)
    else:
        linear = a * k.square() - b * k.pow(4)
        nonlinear_coeff = c if not structural else torch.zeros_like(c)
    mask = (torch.arange(k.shape[-1], device=state.device) < n / 3).to(state.dtype)[None]
    phase = torch.arange(n, device=state.device, dtype=state.dtype) * (2 * math.pi / n)
    forcing = 0.4 * torch.sin(phase) if name == "burgers" else (
        0.1 * torch.sin(phase) + 0.05 * torch.cos(2 * phase))
    force_hat = torch.fft.rfft(action * forcing[None], dim=-1)
    h = dt / substeps
    # Coefficients in float64 prevent cancellation errors when training states
    # use float32. Cast afterward; gradients through coefficients are retained.
    z = (h * linear).to(torch.float64)
    E, E2 = z.exp().to(state.dtype), (z / 2).exp().to(state.dtype)
    Q = (h / 2 * _phi(z / 2, 1)).to(state.dtype)
    p1, p2, p3 = (_phi(z, i) for i in (1, 2, 3))
    f1 = (h * (p1 - 3 * p2 + 4 * p3)).to(state.dtype)
    f2 = (h * (p2 - 2 * p3)).to(state.dtype)
    f3 = (h * (-p2 + 4 * p3)).to(state.dtype)

    def nonlinear(v):
        u = torch.fft.irfft(v * mask, n=n, dim=-1)
        adv = (-0.5j * k) * torch.fft.rfft(u.square(), dim=-1)
        return nonlinear_coeff * adv * mask + force_hat

    v = torch.fft.rfft(state[:, 0], dim=-1)
    for _ in range(substeps):
        Nv = nonlinear(v)
        va = E2 * v + Q * Nv
        Na = nonlinear(va)
        vb = E2 * v + Q * Na
        Nb = nonlinear(vb)
        vc = E2 * va + Q * (2 * Nb - Nv)
        Nc = nonlinear(vc)
        v = E * v + f1 * Nv + 2 * f2 * (Na + Nb) + f3 * Nc
    return torch.fft.irfft(v, n=n, dim=-1)[:, None]


def _linear_transport_step(state: torch.Tensor, action: torch.Tensor,
                           params: torch.Tensor, structural: bool,
                           dt: float, length: float) -> torch.Tensor:
    """Exact affine transition of the real Fourier collocation system."""
    n = state.shape[-1]
    k = _wave_numbers(state, length)[None]
    # A first derivative of the Nyquist cosine is zero on an even real grid.
    # Damping of that mode still uses its nonzero second derivative.
    first_k = k.clone()
    first_k[..., -1] = 0
    diffusion, speed, drag = (params[:, i:i + 1] for i in range(3))
    linear = -diffusion * k.square() - drag
    if not structural:
        linear = linear - 1j * speed * first_k
    z = (dt * linear).to(torch.complex128)
    dtype = torch.complex128 if state.dtype == torch.float64 else torch.complex64
    E = z.exp().to(dtype)
    Q = (dt * _phi(z, 1)).to(dtype)
    phase = torch.arange(n, device=state.device, dtype=state.dtype) * (2 * math.pi / n)
    forcing = 0.4 * torch.sin(phase) + 0.15 * torch.cos(3 * phase)
    force_hat = torch.fft.rfft(action * forcing[None], dim=-1)
    updated = E * torch.fft.rfft(state[:, 0], dim=-1) + Q * force_hat
    return torch.fft.irfft(updated, n=n, dim=-1)[:, None]


def _reaction_diffusion_step(name: str, state: torch.Tensor,
                             action: torch.Tensor, params: torch.Tensor,
                             structural: bool, substeps: int,
                             dt: float, length: float) -> torch.Tensor:
    """ETDRK4 for a declared Fourier-collocation reaction-diffusion ODE.

    Cubic products are evaluated on the physical grid without truncation.
    The synthetic target is this fixed-grid ODE, not a continuum solution.
    No positivity projection or state clipping is applied.
    """
    n = state.shape[-1]
    k2 = _wave_numbers(state, length).square()[None, None]
    a, b, c = (params[:, i:i + 1, None] for i in range(3))
    phase = torch.arange(n, device=state.device, dtype=state.dtype) * (2 * math.pi / n)
    if name == "allen_cahn":
        reaction_rate = b if not structural else torch.zeros_like(b)
        linear = -a * k2 + reaction_rate
        forcing = c * action[..., None] * torch.sin(phase)[None, None]

        def nonlinear(v):
            y = torch.fft.irfft(v, n=n, dim=-1)
            return torch.fft.rfft(-reaction_rate * y.pow(3) + forcing, dim=-1)
    else:
        diffusivity = torch.cat((a, 0.5 * a), dim=1)
        linear = -diffusivity * k2
        # Spatially modulated positive feed drives this open chemical system.
        # With the documented action range [-1,1], feed is >= 0.65*b.
        feed = b * (1 + 0.35 * action[..., None] * torch.sin(phase)[None, None])

        def nonlinear(v):
            y = torch.fft.irfft(v, n=n, dim=-1)
            substrate, product = y[:, 0:1], y[:, 1:2]
            reaction = substrate * product.square()
            if structural:
                reaction = torch.zeros_like(reaction)
            rhs = torch.cat((-reaction + feed * (1 - substrate),
                             reaction - (feed + c) * product), dim=1)
            return torch.fft.rfft(rhs, dim=-1)

    h = dt / substeps
    z = (h * linear).to(torch.float64)
    E, E2 = z.exp().to(state.dtype), (z / 2).exp().to(state.dtype)
    Q = (h / 2 * _phi(z / 2, 1)).to(state.dtype)
    p1, p2, p3 = (_phi(z, order) for order in (1, 2, 3))
    f1 = (h * (p1 - 3 * p2 + 4 * p3)).to(state.dtype)
    f2 = (h * (p2 - 2 * p3)).to(state.dtype)
    f3 = (h * (-p2 + 4 * p3)).to(state.dtype)
    v = torch.fft.rfft(state, dim=-1)
    for _ in range(substeps):
        Nv = nonlinear(v)
        va = E2 * v + Q * Nv
        Na = nonlinear(va)
        vb = E2 * v + Q * Na
        Nb = nonlinear(vb)
        vc = E2 * va + Q * (2 * Nb - Nv)
        Nc = nonlinear(vc)
        v = E * v + f1 * Nv + 2 * f2 * (Na + Nb) + f3 * Nc
    return torch.fft.irfft(v, n=n, dim=-1)


def step(name: str, state: torch.Tensor, action: torch.Tensor,
         params: torch.Tensor, mismatch: str = "correct",
         fidelity: str = "coarse") -> torch.Tensor:
    """Advance one observation interval; params must already be adjusted.

    No coefficient perturbation is performed here, avoiding double adjustment.
    Only mismatch='structural' changes the equations. No state is clipped.
    """
    s = spec(name, state.shape[-1] if state.shape[-1] > 1 else 32)
    if fidelity not in _FIDELITY:
        raise ValueError(f"Unknown fidelity {fidelity!r}")
    if mismatch not in ("correct", "coefficient", "structural"):
        raise ValueError(f"Unknown mismatch {mismatch!r}")
    if state.ndim != 3 or tuple(state.shape[1:]) != (s["channels"], s["grid"]):
        raise ValueError(f"Invalid state shape {tuple(state.shape)} for {name}")
    if params.shape != (state.shape[0], 3) or action.shape != (state.shape[0], 1):
        raise ValueError("Expected params [B,3], action [B,1] with matching batch")
    structural = mismatch == "structural"
    substeps = _FIDELITY[fidelity]
    p = params.to(device=state.device, dtype=state.dtype)
    u = action.to(device=state.device, dtype=state.dtype)
    if name in ("burgers", "ks"):
        # Resolve explicit nonlinear advection increasingly at larger grids.
        substeps *= max(1, math.ceil(s["grid"] / 64))
        return _spectral_step(name, state, u, p, structural,
                              substeps, s["dt"], s["length"])
    if name == "advection_diffusion":
        return _linear_transport_step(state, u, p, structural, s["dt"], s["length"])
    if name in ("allen_cahn", "gray_scott"):
        return _reaction_diffusion_step(name, state, u, p, structural,
                                        substeps, s["dt"], s["length"])
    if name == "reactor":
        def rhs(y):
            concentration, temperature = y[:, 0, 0], y[:, 1, 0]
            rate = p[:, 0] * concentration
            if not structural:
                rate = rate * torch.exp(p[:, 1] * temperature / (1 + 0.3 * temperature))
            dc = 1 - concentration - rate
            dtemp = rate - temperature + p[:, 2] * (u[:, 0] - temperature)
            return torch.stack((dc, dtemp), dim=1)[..., None]
    elif name == "pendulum":
        def rhs(y):
            angle, velocity = y[:, 0, 0], y[:, 1, 0]
            damping = 0 if structural else p[:, 1] * velocity
            acceleration = -p[:, 0] * torch.sin(angle) - damping + p[:, 2] * u[:, 0]
            return torch.stack((velocity, acceleration), dim=1)[..., None]
    else:
        # Fixed resolution scaling keeps RK4 inside its linear stability region
        # on the declared ID/OOD/approximate coefficient ranges.
        substeps *= max(1, math.ceil(s["grid"] / 32))
        k2 = _wave_numbers(state, s["length"]).square()[None]
        phase = torch.arange(s["grid"], device=state.device,
                             dtype=state.dtype) * (2 * math.pi / s["grid"])
        forcing = 0.5 * u * torch.sin(s.get("forcing_mode", 1) * phase)[None]
        def rhs(y):
            displacement, velocity = y[:, 0], y[:, 1]
            laplacian = torch.fft.irfft(-k2 * torch.fft.rfft(displacement), n=s["grid"])
            damping = 0 if structural else p[:, 1:2] * velocity
            acceleration = p[:, 0:1].square() * laplacian - damping - p[:, 2:3] * displacement + forcing
            return torch.stack((velocity, acceleration), dim=1)
    out = state
    for _ in range(substeps):
        out = _rk4(out, rhs, s["dt"] / substeps)
    return out


def simulate(name: str, params: np.ndarray, initial: np.ndarray,
             actions: np.ndarray, fidelity: str = "fine") -> np.ndarray:
    """CPU float64 numerical reference, explicit failure on nonfinite states."""
    y = torch.as_tensor(initial, dtype=torch.float64)
    p = torch.as_tensor(params, dtype=torch.float64)
    drives = torch.as_tensor(actions, dtype=torch.float64)
    states = [y.detach().cpu().numpy().copy()]
    with torch.no_grad():
        for t in range(drives.shape[1]):
            y = step(name, y, drives[:, t], p, fidelity=fidelity)
            if not torch.isfinite(y).all():
                raise FloatingPointError(f"{name} nonfinite reference at transition {t}")
            states.append(y.cpu().numpy().copy())
    return np.stack(states, axis=1)


def numerical_audit(name: str, grid: int = 32) -> dict[str, Any]:
    """Check time refinement and numerical invariants on deterministic probes.

    This audits the declared finite-dimensional simulator. It does not establish
    spatial convergence to the continuum PDE or validate a real physical device.
    """
    rng = np.random.default_rng(920314)
    params = np.concatenate([sample_params(name, 2, rng), sample_params(name, 2, rng, ood=True)])
    initial = initial_states(name, 4, grid, rng)
    burn_steps = {"ks": 128, "allen_cahn": 24, "gray_scott": 24}.get(name, 0)
    if burn_steps:
        # Audit the richer evolved regime as well as the smooth startup.
        initial = simulate(name, params, initial,
                           make_actions(name, 4, burn_steps, rng), "fine")[:, -1]
    actions = make_actions(name, 4, 24, rng)
    low, high = (-0.05, 0.25) if name == "reactor" else ((-2.0, 2.0) if name == "pendulum" else (-1.0, 1.0))
    actions[2, :, 0], actions[3, :, 0] = low, high
    runs = {f: simulate(name, params, initial, actions, f) for f in _FIDELITY}
    reference = runs["check"]
    scale = max(float(np.sqrt(np.mean(reference ** 2))), 0.1)
    ec = float(np.sqrt(np.mean((runs["coarse"] - reference) ** 2)) / scale)
    ef = float(np.sqrt(np.mean((runs["fine"] - reference) ** 2)) / scale)
    tolerance = 2e-4 if name == "ks" else 2e-5
    result = {
        "system": name, "grid": spec(name, grid)["grid"], "trajectories": 4,
        "steps": 24, "burn_in_steps": burn_steps,
        "includes_ood_and_extreme_drives": True,
        "dtype": "float64", "all_finite": True,
        "coarse_vs_check_relative_rmse": ec, "fine_vs_check_relative_rmse": ef,
        "fine_relative_rmse_tolerance": tolerance,
        "refinement_improves": bool(ef <= ec * 0.5 + 1e-11),
        "temporal_convergence_pass": bool(ef < tolerance and ef <= ec * 0.5 + 1e-11),
        "scope": "Temporal convergence of fixed-grid synthetic reference; no continuum or real-device validation.",
    }
    if name in ("pendulum", "wave", "wave_high", "burgers", "advection_diffusion", "allen_cahn"):
        # The drive-free systems should dissipate their declared energy.
        s = simulate(name, params, initial, np.zeros((4, 24, 1)), "fine")
        if name == "pendulum":
            energy = 0.5 * s[:, :, 1, 0] ** 2 + params[:, 0, None] * (1 - np.cos(s[:, :, 0, 0]))
        elif name in ("burgers", "advection_diffusion"):
            energy = 0.5 * np.mean(s[:, :, 0] ** 2, axis=-1)
        elif name == "allen_cahn":
            k = 2 * np.pi * np.fft.rfftfreq(grid, d=spec(name, grid)["length"] / grid)
            lap = np.fft.irfft(-k ** 2 * np.fft.rfft(s[:, :, 0], axis=-1), n=grid, axis=-1)
            energy = (0.5 * params[:, 0, None] * np.mean(-s[:, :, 0] * lap, axis=-1) +
                      0.25 * params[:, 1, None] * np.mean((s[:, :, 0] ** 2 - 1) ** 2, axis=-1))
        else:
            k = 2 * np.pi * np.fft.rfftfreq(grid, d=spec(name, grid)["length"] / grid)
            dx = np.fft.irfft(1j * k * np.fft.rfft(s[:, :, 0], axis=-1), n=grid, axis=-1)
            energy = 0.5 * np.mean(s[:, :, 1] ** 2 + params[:, 0, None, None] ** 2 * dx ** 2
                                         + params[:, 2, None, None] * s[:, :, 0] ** 2, axis=-1)
        increase = float(np.max(np.diff(energy, axis=1)))
        result["maximum_unforced_energy_increment"] = increase
        result["energy_dissipation_pass"] = bool(increase <= 1e-9)
    if name == "reactor":
        result["minimum_concentration"] = float(reference[:, :, 0].min())
        result["positive_concentration_pass"] = bool(reference[:, :, 0].min() > 0)
    if name == "gray_scott":
        result["minimum_concentration"] = float(reference.min())
        result["positive_concentration_pass"] = bool(reference.min() >= -1e-10)
    result["passed"] = bool(result["temporal_convergence_pass"] and
                            result.get("energy_dissipation_pass", True) and
                            result.get("positive_concentration_pass", True))
    return result
