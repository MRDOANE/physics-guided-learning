"""Pure-NumPy float64 port of the archived controlled dynamical systems.

This module intentionally uses float64 for states and coefficients. It preserves
the reference equations, fixed-grid products, masks, ETDRK4 coefficients, forcing,
and fidelity-dependent substeps. It supplies no automatic differentiation.

All spectral systems use a periodic, equispaced grid without a duplicated endpoint.
See docs/PHYSICS.md for equations, discretization and the limits of the reference.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

_SYSTEMS = {
    "cahn_hilliard": dict(channels=1, grid=64, dt=0.05, length=2 * math.pi,
               param_names=["interfacial_stiffness", "reaction_scale", "forcing_strength"],
               param_center=[0.10, 0.50, 0.05]),
    "fitzhugh_nagumo": dict(channels=2, grid=64, dt=0.10, length=2 * math.pi,
               param_names=["diffusivity", "recovery_rate", "forcing_strength"],
               param_center=[0.03, 0.12, 0.20]),
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
                 "advection_diffusion": 0.6, "allen_cahn": 0.4,
                 "cahn_hilliard": 0.25, "fitzhugh_nagumo": 0.6}[name]
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
    if name in ("cahn_hilliard", "fitzhugh_nagumo"):
        out[:, 0] += rng.uniform(-0.15, 0.15, (count, 1))
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


def approx_params(name: str, params: np.ndarray, mismatch: str) -> np.ndarray:
    """Return the declared nominal parameters; step never reapplies this bias."""
    spec(name)
    params = np.asarray(params, dtype=np.float64)
    if mismatch in ("correct", "structural"):
        return params.copy()
    if mismatch == "coefficient":
        return params * np.asarray([0.70, 1.30, 1.40], dtype=np.float64)
    raise ValueError(f"Unknown mismatch {mismatch!r}")


def _rk4(state: np.ndarray, rhs, h: float) -> np.ndarray:
    k1 = rhs(state)
    k2 = rhs(state + (h / 2) * k1)
    k3 = rhs(state + (h / 2) * k2)
    k4 = rhs(state + h * k3)
    return state + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def _wave_numbers(state: np.ndarray, length: float) -> np.ndarray:
    return 2 * math.pi * np.fft.rfftfreq(state.shape[-1], d=length / state.shape[-1])


def _phi(z: np.ndarray, order: int) -> np.ndarray:
    """Same order-10 Taylor branch and 0.2 cutoff as the archived solver."""
    small = np.abs(z) < 0.2
    safe = np.where(small, np.ones_like(z), z)
    em = np.expm1(z)
    if order == 1:
        direct = em / safe
    elif order == 2:
        direct = (em - z) / safe ** 2
    else:
        direct = (em - z - z ** 2 / 2) / safe ** 3
    series = np.full_like(z, 1.0 / math.factorial(order + 10))
    for j in range(9, -1, -1):
        series = series * z + 1.0 / math.factorial(order + j)
    return np.where(small, series, direct)


def _etd_evolve(v: np.ndarray, linear: np.ndarray, nonlinear,
                dt: float, substeps: int) -> np.ndarray:
    """Cox--Matthews ETDRK4; operation order follows the archived source."""
    h = dt / substeps
    z = np.asarray(h * linear, dtype=np.float64)
    E, E2 = np.exp(z), np.exp(z / 2)
    Q = h / 2 * _phi(z / 2, 1)
    p1, p2, p3 = (_phi(z, i) for i in (1, 2, 3))
    f1 = h * (p1 - 3 * p2 + 4 * p3)
    f2 = h * (p2 - 2 * p3)
    f3 = h * (-p2 + 4 * p3)
    for _ in range(substeps):
        Nv = nonlinear(v)
        va = E2 * v + Q * Nv
        Na = nonlinear(va)
        vb = E2 * v + Q * Na
        Nb = nonlinear(vb)
        vc = E2 * va + Q * (2 * Nb - Nv)
        Nc = nonlinear(vc)
        v = E * v + f1 * Nv + 2 * f2 * (Na + Nb) + f3 * Nc
    return v


def _spectral_step(name: str, state: np.ndarray, action: np.ndarray,
                   params: np.ndarray, structural: bool, substeps: int,
                   dt: float, length: float) -> np.ndarray:
    n = state.shape[-1]
    k = _wave_numbers(state, length)[None, :]
    a, b, c = (params[:, i:i + 1] for i in range(3))
    if name == "burgers":
        linear = -a * k ** 2 - c
        nonlinear_coeff = b if not structural else np.zeros_like(b)
    else:
        linear = a * k ** 2 - b * k ** 4
        nonlinear_coeff = c if not structural else np.zeros_like(c)
    mask = (np.arange(k.shape[-1]) < n / 3).astype(np.float64)[None]
    phase = np.arange(n, dtype=np.float64) * (2 * math.pi / n)
    forcing = 0.4 * np.sin(phase) if name == "burgers" else (
        0.1 * np.sin(phase) + 0.05 * np.cos(2 * phase))
    force_hat = np.fft.rfft(action * forcing[None], axis=-1)

    def nonlinear(v):
        u = np.fft.irfft(v * mask, n=n, axis=-1)
        adv = (-0.5j * k) * np.fft.rfft(u ** 2, axis=-1)
        return nonlinear_coeff * adv * mask + force_hat

    v = _etd_evolve(np.fft.rfft(state[:, 0], axis=-1), linear,
                    nonlinear, dt, substeps)
    return np.fft.irfft(v, n=n, axis=-1)[:, None]


def _linear_transport_step(state: np.ndarray, action: np.ndarray,
                           params: np.ndarray, structural: bool,
                           dt: float, length: float) -> np.ndarray:
    n = state.shape[-1]
    k = _wave_numbers(state, length)[None]
    first_k = k.copy()
    first_k[..., -1] = 0
    diffusion, speed, drag = (params[:, i:i + 1] for i in range(3))
    linear = -diffusion * k ** 2 - drag
    if not structural:
        linear = linear - 1j * speed * first_k
    z = np.asarray(dt * linear, dtype=np.complex128)
    E = np.exp(z)
    Q = dt * _phi(z, 1)
    phase = np.arange(n, dtype=np.float64) * (2 * math.pi / n)
    forcing = 0.4 * np.sin(phase) + 0.15 * np.cos(3 * phase)
    force_hat = np.fft.rfft(action * forcing[None], axis=-1)
    updated = E * np.fft.rfft(state[:, 0], axis=-1) + Q * force_hat
    return np.fft.irfft(updated, n=n, axis=-1)[:, None]


def _reaction_diffusion_step(name: str, state: np.ndarray,
                             action: np.ndarray, params: np.ndarray,
                             structural: bool, substeps: int,
                             dt: float, length: float) -> np.ndarray:
    """Original fixed-grid cubic products; no truncation or positivity clips."""
    n = state.shape[-1]
    k2 = (_wave_numbers(state, length) ** 2)[None, None]
    a, b, c = (params[:, i:i + 1, None] for i in range(3))
    phase = np.arange(n, dtype=np.float64) * (2 * math.pi / n)
    if name == "allen_cahn":
        reaction_rate = b if not structural else np.zeros_like(b)
        linear = -a * k2 + reaction_rate
        forcing = c * action[..., None] * np.sin(phase)[None, None]

        def nonlinear(v):
            y = np.fft.irfft(v, n=n, axis=-1)
            return np.fft.rfft(-reaction_rate * y ** 3 + forcing, axis=-1)
    else:
        diffusivity = np.concatenate((a, 0.5 * a), axis=1)
        linear = -diffusivity * k2
        feed = b * (1 + 0.35 * action[..., None] * np.sin(phase)[None, None])

        def nonlinear(v):
            y = np.fft.irfft(v, n=n, axis=-1)
            substrate, product = y[:, 0:1], y[:, 1:2]
            reaction = substrate * product ** 2
            if structural:
                reaction = np.zeros_like(reaction)
            rhs = np.concatenate((-reaction + feed * (1 - substrate),
                                  reaction - (feed + c) * product), axis=1)
            return np.fft.rfft(rhs, axis=-1)

    v = _etd_evolve(np.fft.rfft(state, axis=-1), linear, nonlinear, dt, substeps)
    return np.fft.irfft(v, n=n, axis=-1)


def _new_equation_step(name: str, state: np.ndarray, action: np.ndarray,
                       params: np.ndarray, structural: bool, substeps: int,
                       dt: float, length: float) -> np.ndarray:
    n = state.shape[-1]
    k2 = (_wave_numbers(state, length) ** 2)[None, None]
    a, b, c = (params[:, i:i + 1, None] for i in range(3))
    phase = np.arange(n, dtype=np.float64) * (2 * math.pi / n)
    forcing = c * action[..., None] * np.sin(phase)[None, None]
    if name == "cahn_hilliard":
        linear = b * k2 - a * k2 ** 2

        def nonlinear(v):
            y = np.fft.irfft(v, n=n, axis=-1)
            reaction = (np.zeros_like(v) if structural else
                        -b * k2 * np.fft.rfft(y ** 3, axis=-1))
            f = np.fft.rfft(forcing, axis=-1)
            f[..., 0] = 0
            return reaction + f
    else:
        linear = np.concatenate((1 - a * k2,
                                 np.broadcast_to(-0.8 * b, (len(state), 1, k2.shape[-1]))), axis=1)

        def nonlinear(v):
            y = np.fft.irfft(v, n=n, axis=-1)
            u, w = y[:, 0:1], y[:, 1:2]
            feedback = np.zeros_like(w) if structural else w
            return np.fft.rfft(np.concatenate((-u ** 3 / 3 - feedback + forcing,
                                              b * (u + 0.7)), axis=1), axis=-1)

    v = _etd_evolve(np.fft.rfft(state, axis=-1), linear, nonlinear, dt, substeps)
    return np.fft.irfft(v, n=n, axis=-1)


def step(name: str, state: np.ndarray, action: np.ndarray,
         params: np.ndarray, mismatch: str = "correct",
         fidelity: str = "coarse") -> np.ndarray:
    """Advance one interval in float64, with already-adjusted parameters.

    Only 'structural' changes the equations. 'Coefficient' does not apply the
    bias a second time. Shapes are state [B,C,G], action [B,1], params [B,3].
    """
    state = np.asarray(state, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    if state.ndim != 3:
        raise ValueError(f"Invalid state shape {tuple(state.shape)} for {name}")
    s = spec(name, state.shape[-1] if state.shape[-1] > 1 else 32)
    if fidelity not in _FIDELITY:
        raise ValueError(f"Unknown fidelity {fidelity!r}")
    if mismatch not in ("correct", "coefficient", "structural"):
        raise ValueError(f"Unknown mismatch {mismatch!r}")
    if tuple(state.shape[1:]) != (s["channels"], s["grid"]):
        raise ValueError(f"Invalid state shape {tuple(state.shape)} for {name}")
    if params.shape != (state.shape[0], 3) or action.shape != (state.shape[0], 1):
        raise ValueError("Expected params [B,3], action [B,1] with matching batch")
    structural = mismatch == "structural"
    substeps = _FIDELITY[fidelity]
    p, u = params, action
    if name in ("cahn_hilliard", "fitzhugh_nagumo"):
        return _new_equation_step(name, state, u, p, structural, substeps, s["dt"], s["length"])
    if name in ("burgers", "ks"):
        substeps *= max(1, math.ceil(s["grid"] / 64))
        return _spectral_step(name, state, u, p, structural, substeps, s["dt"], s["length"])
    if name == "advection_diffusion":
        return _linear_transport_step(state, u, p, structural, s["dt"], s["length"])
    if name in ("allen_cahn", "gray_scott"):
        return _reaction_diffusion_step(name, state, u, p, structural, substeps, s["dt"], s["length"])
    if name == "reactor":
        def rhs(y):
            concentration, temperature = y[:, 0, 0], y[:, 1, 0]
            rate = p[:, 0] * concentration
            if not structural:
                rate = rate * np.exp(p[:, 1] * temperature / (1 + 0.3 * temperature))
            dc = 1 - concentration - rate
            dtemp = rate - temperature + p[:, 2] * (u[:, 0] - temperature)
            return np.stack((dc, dtemp), axis=1)[..., None]
    elif name == "pendulum":
        def rhs(y):
            angle, velocity = y[:, 0, 0], y[:, 1, 0]
            damping = 0 if structural else p[:, 1] * velocity
            acceleration = -p[:, 0] * np.sin(angle) - damping + p[:, 2] * u[:, 0]
            return np.stack((velocity, acceleration), axis=1)[..., None]
    else:
        substeps *= max(1, math.ceil(s["grid"] / 32))
        k2 = (_wave_numbers(state, s["length"]) ** 2)[None]
        phase = np.arange(s["grid"], dtype=np.float64) * (2 * math.pi / s["grid"])
        forcing = 0.5 * u * np.sin(s.get("forcing_mode", 1) * phase)[None]

        def rhs(y):
            displacement, velocity = y[:, 0], y[:, 1]
            laplacian = np.fft.irfft(-k2 * np.fft.rfft(displacement), n=s["grid"])
            damping = 0 if structural else p[:, 1:2] * velocity
            acceleration = p[:, 0:1] ** 2 * laplacian - damping - p[:, 2:3] * displacement + forcing
            return np.stack((velocity, acceleration), axis=1)

    out = state
    for _ in range(substeps):
        out = _rk4(out, rhs, s["dt"] / substeps)
    return out


def simulate(name: str, params: np.ndarray, initial: np.ndarray,
             actions: np.ndarray, fidelity: str = "fine") -> np.ndarray:
    """CPU float64 reference; fail explicitly on a nonfinite transition."""
    y = np.array(initial, dtype=np.float64, copy=True)
    p = np.asarray(params, dtype=np.float64)
    drives = np.asarray(actions, dtype=np.float64)
    states = [y.copy()]
    for t in range(drives.shape[1]):
        y = step(name, y, drives[:, t], p, fidelity=fidelity)
        if not np.isfinite(y).all():
            raise FloatingPointError(f"{name} nonfinite reference at transition {t}")
        states.append(y.copy())
    return np.stack(states, axis=1)
