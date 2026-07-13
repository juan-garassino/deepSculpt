"""Procedural RGBA palettes for color latent diffusion.

Diffusion wants continuous channels, so color sculptures are represented as an
RGBA voxel field — [alpha, R, G, B] (alpha first, so channel 0 stays "presence"
and every threshold-channel-0 convention downstream keeps working). Instead of
12 flat hex colors, each sculpture gets its own procedurally-generated pastel
scheme with a soft vertical gradient, seeded deterministically by the sample
index so the color is stable across epochs, workers, and resumed cloud slices.

This module is the single source of truth for the element→color mapping.
`scripts/render_walk.py` / `scripts/walk_viewer.py` keep dependency-light local
copies of CLASS_PALETTE (they run outside the package) — keep them in sync.

Determinism contract (load-bearing — the latent shift/scale resume assertion
depends on it): identical (mode, seed, version, index) always yields identical
RGBA. A dedicated CPU torch.Generator is seeded by an explicit integer mix
(never hash(), never the global RNG), and every sample draws the same fixed
number of values in the same order regardless of mode or which classes appear.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

# Canonical element palette (class index -> hex). Mirrors shodhan.py classes:
# COL, SLAB, SCREEN, PIPE_R, PIPE_B, PIPE_Y, VOL, EDGE = 1..8;
# WALL_R, WALL_B, WALL_Y, WALL_G = 9..12.
CLASS_PALETTE = {
    1: "#d9d4cc", 2: "#efe9df", 3: "#c8c2b8", 4: "#c0392b",
    5: "#2e6da4", 6: "#d4a017", 7: "#b9b2a6", 8: "#8f887c",
    9: "#c0392b", 10: "#2e6da4", 11: "#d4a017", 12: "#3d8b5f",
}
NUM_CLASSES = 12
# Concrete structure classes keep low saturation even in 'bold'; the coloured
# pipes/walls carry the per-sample hue.
NEUTRAL_CLASSES = (1, 2, 3, 7, 8)
ACCENT_CLASSES = (4, 5, 6, 9, 10, 11, 12)
PALETTE_VERSION = 1
_SEED_MASK = 0x7FFFFFFFFFFFFFFF


def _hex_to_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def _rgb_to_hsv(rgb: torch.Tensor) -> torch.Tensor:
    """rgb (..., 3) in [0,1] -> hsv (..., 3)."""
    r, g, b = rgb.unbind(-1)
    mx, _ = rgb.max(-1)
    mn, _ = rgb.min(-1)
    diff = (mx - mn).clamp_min(1e-8)
    h = torch.zeros_like(mx)
    mask = mx == r
    h = torch.where(mask, ((g - b) / diff) % 6.0, h)
    mask = mx == g
    h = torch.where(mask, (b - r) / diff + 2.0, h)
    mask = mx == b
    h = torch.where(mask, (r - g) / diff + 4.0, h)
    h = (h / 6.0) % 1.0
    s = torch.where(mx > 0, (mx - mn) / mx.clamp_min(1e-8), torch.zeros_like(mx))
    return torch.stack([h, s, mx], dim=-1)


def _hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
    """hsv (..., 3) -> rgb (..., 3), all in [0,1]."""
    h, s, v = hsv.unbind(-1)
    i = (h * 6.0).floor()
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i.long() % 6
    r = torch.stack([v, q, p, p, t, v], dim=-1)
    g = torch.stack([t, v, v, q, p, p], dim=-1)
    b = torch.stack([p, p, t, v, v, q], dim=-1)
    idx = i.unsqueeze(-1)
    return torch.cat([
        r.gather(-1, idx), g.gather(-1, idx), b.gather(-1, idx)
    ], dim=-1)


# Base HSV per class (index 0 = empty, unused). Shape [13, 3].
_BASE_HSV = _rgb_to_hsv(torch.tensor(
    [[0.0, 0.0, 0.0]] + [list(_hex_to_rgb(CLASS_PALETTE[c])) for c in range(1, 13)]
))
_ACCENT_MASK = torch.zeros(13, dtype=torch.bool)
for _c in ACCENT_CLASSES:
    _ACCENT_MASK[_c] = True


@dataclass
class PaletteConfig:
    mode: str = "subtle"          # flat | subtle | bold
    seed: int = 0
    version: int = PALETTE_VERSION


def _sample_endpoints(index: int, cfg: PaletteConfig) -> torch.Tensor:
    """Per-class RGB endpoints for one sample: [13, 2, 3] (class, {bottom,top}, rgb).

    Fixed draw layout — global rotation then a [12, 2, 3] jitter table — so the
    RNG stream never depends on mode or content.
    """
    seed = (cfg.seed * 1_000_003 + int(index)) & _SEED_MASK
    g = torch.Generator().manual_seed(seed)
    global_rot = torch.rand(1, generator=g).item()
    tbl = torch.rand(NUM_CLASSES, 2, 3, generator=g)  # [class, endpoint, {hue,sat,val}]

    base = _BASE_HSV[1:].unsqueeze(1).expand(NUM_CLASSES, 2, 3).clone()  # [12,2,3]
    h0, s0, v0 = base.unbind(-1)

    if cfg.mode == "flat":
        rgb = _hsv_to_rgb(base)  # exact base colour, both endpoints identical
        out = torch.zeros(13, 2, 3)
        out[1:] = rgb
        return out

    hue_j = (tbl[..., 0] - 0.5) * 0.08          # +-0.04
    sat = (s0 * (0.45 + 0.35 * tbl[..., 1])).clamp(0.05, 0.55)
    val = 0.80 + 0.15 * tbl[..., 2]             # pastel brightness

    if cfg.mode == "bold":
        accent = _ACCENT_MASK[1:].view(NUM_CLASSES, 1)
        hue = torch.where(accent, (h0 + global_rot + hue_j) % 1.0, (h0 + hue_j) % 1.0)
        sat = torch.where(accent, (0.35 + 0.35 * tbl[..., 1]).clamp(0.05, 0.75), sat)
    else:  # subtle
        hue = (h0 + hue_j) % 1.0

    hsv = torch.stack([hue, sat, val], dim=-1)  # [12,2,3]
    out = torch.zeros(13, 2, 3)
    out[1:] = _hsv_to_rgb(hsv)
    return out


def build_rgba(
    structure: torch.Tensor,
    colors: torch.Tensor,
    indices: torch.Tensor,
    cfg: PaletteConfig,
) -> torch.Tensor:
    """[B,4,D,H,W] RGBA float in [0,1]. alpha = structure; RGB from the
    per-sample pastel scheme with a vertical (last-axis) gradient; empty voxels
    are neutral 0.5 so masked losses see no meaningful colour there."""
    if structure.dim() == 5:
        structure = structure.squeeze(1)
    if colors.dim() == 5:
        colors = colors.squeeze(1)
    dev = structure.device
    B, D, H, W = structure.shape
    idx = colors.long().clamp(0, NUM_CLASSES)  # [B,D,H,W]

    # Per-sample endpoint tables (CPU generator, trivial cost), then to device.
    ends = torch.stack([_sample_endpoints(int(i), cfg) for i in indices]).to(dev)  # [B,13,2,3]
    b_ar = torch.arange(B, device=dev)[:, None, None, None]
    bottom = ends[b_ar, idx, 0]  # [B,D,H,W,3]
    top = ends[b_ar, idx, 1]

    ramp = torch.linspace(0.0, 1.0, W, device=dev).view(1, 1, 1, W, 1)
    rgb = bottom * (1.0 - ramp) + top * ramp

    occ = (structure > 0.5).unsqueeze(-1)
    rgb = torch.where(occ, rgb, torch.full_like(rgb, 0.5))

    alpha = structure.float().unsqueeze(-1)
    rgba = torch.cat([alpha, rgb], dim=-1)  # [B,D,H,W,4]
    return rgba.permute(0, 4, 1, 2, 3).contiguous()
