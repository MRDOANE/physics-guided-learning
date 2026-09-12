"""Small state-space predictors with an explicit shared-depth transformer arm.

All predictors receive the same observed states, actions, and approximate
coefficients. Inputs and outputs are normalized by training.py. The residual
head starts at zero, so every neural arm initially predicts persistence.
"""
from __future__ import annotations

import math
import torch
from torch import nn


def _position(length: int, width: int) -> torch.Tensor:
    pos = torch.arange(length, dtype=torch.float32)[:, None]
    rate = torch.exp(torch.arange(0, width, 2, dtype=torch.float32) * (-math.log(10000.0) / width))
    out = torch.zeros(length, width)
    out[:, 0::2] = torch.sin(pos * rate)
    out[:, 1::2] = torch.cos(pos * rate[:out[:, 1::2].shape[1]])
    return out


class StateTransformer(nn.Module):
    def __init__(self, channels, grid, context, width, depth, heads, shared=False):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.grid, self.context, self.depth, self.shared = grid, context, depth, shared
        self.embed = nn.Linear(channels + 4, width)
        self.register_buffer("time_position", _position(context, width)[None, :, None, :])
        self.register_buffer("space_position", _position(grid, width)[None, None, :, :])
        count = 1 if shared else depth
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(width, heads, 2 * width, dropout=0.0,
                                       activation="gelu", batch_first=True, norm_first=True)
            for _ in range(count)
        ])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, channels)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, history, action_history, params):
        b, k, c, n = history.shape
        x = history.permute(0, 1, 3, 2)
        act = action_history[:, :, None, :].expand(b, k, n, 1)
        coefficients = params[:, None, None, :].expand(b, k, n, 3)
        x = self.embed(torch.cat((x, act, coefficients), dim=-1))
        x = x + self.time_position[:, :k] + self.space_position[:, :, :n]
        x = x.reshape(b, k * n, -1)
        # Full attention is only over the observed context; no future state is present.
        for i in range(self.depth):
            x = self.blocks[0 if self.shared else i](x)
        residual = self.head(self.norm(x[:, -n:])).transpose(1, 2)
        return history[:, -1] + residual


class StateMLP(nn.Module):
    def __init__(self, channels, grid, context, width, depth):
        super().__init__()
        self.channels, self.grid = channels, grid
        inputs = context * channels * grid + context + 3
        layers = [nn.Linear(inputs, width), nn.GELU()]
        for _ in range(max(0, depth - 1)):
            layers.extend((nn.Linear(width, width), nn.GELU()))
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(width, channels * grid)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, history, action_history, params):
        b = history.shape[0]
        x = torch.cat((history.reshape(b, -1), action_history.reshape(b, -1), params), dim=1)
        return history[:, -1] + self.head(self.body(x)).reshape(b, self.channels, self.grid)


class SpectralConv1d(nn.Module):
    """Truncated Fourier convolution; modes are fixed, not selected on test data."""
    def __init__(self, width, modes):
        super().__init__()
        self.modes = modes
        self.weight = nn.Parameter(torch.randn(width, width, modes, dtype=torch.cfloat) / width)

    def forward(self, x):
        z = torch.fft.rfft(x, dim=-1)
        out = torch.zeros(x.shape[0], x.shape[1], z.shape[-1], dtype=z.dtype, device=x.device)
        m = min(self.modes, z.shape[-1])
        out[:, :, :m] = torch.einsum("bim,iom->bom", z[:, :, :m], self.weight[:, :, :m])
        return torch.fft.irfft(out, n=x.shape[-1], dim=-1)


class StateFNO(nn.Module):
    """Compact one-dimensional FNO baseline, with explicit spatial position."""
    def __init__(self, channels, grid, context, width, depth, modes=8):
        super().__init__()
        self.lift = nn.Conv1d(context * channels + context + 3 + 1, width, 1)
        self.spectral = nn.ModuleList([SpectralConv1d(width, min(modes, grid // 2 + 1)) for _ in range(depth)])
        self.local = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(depth)])
        self.head = nn.Conv1d(width, channels, 1)
        self.register_buffer("position", torch.linspace(0.0, 1.0, grid)[None, None, :])
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, history, action_history, params):
        b, k, c, n = history.shape
        act = action_history.reshape(b, k, 1).expand(b, k, n)
        coeff = params[:, :, None].expand(b, 3, n)
        pos = self.position.expand(b, 1, n)
        x = self.lift(torch.cat((history.reshape(b, k * c, n), act, coeff, pos), dim=1))
        for spectral, local in zip(self.spectral, self.local):
            x = torch.nn.functional.gelu(spectral(x) + local(x))
        return history[:, -1] + self.head(x)


def build_model(kind: str, spec: dict, cfg: dict) -> nn.Module:
    c, n = int(spec["channels"]), int(spec["grid"])
    k, w, d = (int(cfg[key]) for key in ("context", "width", "depth"))
    if min(c, n, k, w, d) < 1:
        raise ValueError("Model dimensions must be positive")
    if kind in ("transformer", "looped"):
        return StateTransformer(c, n, k, w, d, int(cfg["heads"]), shared=kind == "looped")
    if kind == "mlp":
        return StateMLP(c, n, k, w, d)
    if kind == "fno":
        if n <= 1:
            raise ValueError("FNO baseline requires grid > 1")
        return StateFNO(c, n, k, w, d, int(cfg.get("fno_modes", 8)))
    raise ValueError(f"Unknown model kind {kind!r}")
