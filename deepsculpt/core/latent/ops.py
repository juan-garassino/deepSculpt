"""
Pure latent-space tensor operations for DeepSculpt.

Everything here is I/O-free and works on either GAN z-vectors of shape
(noise_dim,) / (1, noise_dim) or diffusion initial-noise tensors of shape
(1, C, D, H, W) — interpolation treats the whole tensor as one flat vector.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, Sequence

import torch

logger = logging.getLogger(__name__)


def seeded_z(seed: int, noise_dim: int, batch: int = 1) -> torch.Tensor:
    """Reproducible z batch (batch, noise_dim) — CPU generator so the same
    seed gives the same vector regardless of the compute device."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(batch, noise_dim, generator=gen)


def seeded_noise(seed: int, shape: Sequence[int]) -> torch.Tensor:
    """Reproducible noise tensor of arbitrary shape (diffusion init noise)."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(tuple(shape), generator=gen)


def lerp(z1: torch.Tensor, z2: torch.Tensor, steps: int) -> torch.Tensor:
    """Linear interpolation. Returns (steps, *z1.shape) including endpoints."""
    if steps < 2:
        raise ValueError("steps must be >= 2 to include both endpoints")
    alphas = torch.linspace(0.0, 1.0, steps, dtype=z1.dtype)
    return torch.stack([(1 - a) * z1 + a * z2 for a in alphas])


def slerp(z1: torch.Tensor, z2: torch.Tensor, steps: int) -> torch.Tensor:
    """Spherical interpolation — the right interpolant for Gaussian latents
    (preserves vector norm along the path). Falls back to lerp for
    near-parallel vectors. Returns (steps, *z1.shape) including endpoints."""
    if steps < 2:
        raise ValueError("steps must be >= 2 to include both endpoints")
    v1 = z1.flatten().double()
    v2 = z2.flatten().double()
    cos_omega = torch.clamp(
        torch.dot(v1, v2) / (v1.norm() * v2.norm() + 1e-12), -1.0, 1.0
    )
    omega = torch.acos(cos_omega)
    if omega.abs() < 1e-4:
        return lerp(z1, z2, steps)
    sin_omega = torch.sin(omega)
    alphas = torch.linspace(0.0, 1.0, steps, dtype=torch.float64)
    frames = [
        (torch.sin((1 - a) * omega) / sin_omega) * v1
        + (torch.sin(a * omega) / sin_omega) * v2
        for a in alphas
    ]
    return torch.stack(frames).to(z1.dtype).reshape(steps, *z1.shape)


def walk_path(
    anchors: torch.Tensor,
    steps_per_segment: int,
    mode: str = "slerp",
    closed: bool = False,
) -> torch.Tensor:
    """Multi-anchor walk through latent space.

    anchors: (num_anchors, *z_shape), num_anchors >= 2.
    Returns (total_steps, *z_shape); consecutive segments share their joint
    frame so the path has no duplicated frames.
    """
    if anchors.shape[0] < 2:
        raise ValueError("walk_path needs at least 2 anchors")
    interp = slerp if mode == "slerp" else lerp
    points = list(anchors)
    if closed:
        points.append(anchors[0])
    segments = []
    for a, b in zip(points[:-1], points[1:]):
        seg = interp(a, b, steps_per_segment)
        if segments:
            seg = seg[1:]  # joint frame already emitted by previous segment
        segments.append(seg)
    return torch.cat(segments)


def traverse_dimension(
    z_base: torch.Tensor,
    dim: int,
    steps: int,
    sigma_range: float = 3.0,
) -> torch.Tensor:
    """Vary one latent coordinate across ±sigma_range, all else fixed.
    z_base: (noise_dim,) or (1, noise_dim). Returns (steps, *z_base.shape)."""
    flat = z_base.flatten()
    if not 0 <= dim < flat.numel():
        raise ValueError(f"dim {dim} out of range for latent size {flat.numel()}")
    values = torch.linspace(-sigma_range, sigma_range, steps, dtype=z_base.dtype)
    out = z_base.unsqueeze(0).repeat(steps, *([1] * z_base.dim())).clone()
    out.flatten(1)[:, dim] = values
    return out


_ARITH_EXPR = re.compile(r"\s*[+-]?\s*\w+(\s*[+-]\s*\w+)*\s*")
_ARITH_TOKEN = re.compile(r"([+-])?\s*([A-Za-z_]\w*)")


def latent_arithmetic(zs: Dict[str, torch.Tensor], expr: str) -> torch.Tensor:
    """Evaluate a +/- expression over named latents, e.g. "a - b + c"."""
    if not _ARITH_EXPR.fullmatch(expr):
        raise ValueError(f"cannot parse latent expression: {expr!r}")
    result = None
    for sign, name in _ARITH_TOKEN.findall(expr):
        if name not in zs:
            raise KeyError(f"unknown latent {name!r}; available: {sorted(zs)}")
        term = -zs[name] if sign == "-" else zs[name]
        result = term if result is None else result + term
    return result


def batched_generate(
    generator: torch.nn.Module,
    zs: torch.Tensor,
    batch_size: int = 8,
    device: str = "cpu",
) -> torch.Tensor:
    """Run the generator over (N, noise_dim) latents in minibatches.
    Returns volumes (N, C, D, H, W) on CPU."""
    generator.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, zs.shape[0], batch_size):
            batch = zs[i : i + batch_size].to(device)
            outs.append(generator(batch).cpu())
    return torch.cat(outs)
