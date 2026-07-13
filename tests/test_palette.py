"""Determinism and correctness tests for the RGBA palette (color latent diffusion).

Determinism is load-bearing: the latent shift/scale resume assertion in the
diffusion trainer relies on identical (mode, seed, index) -> identical RGBA
across slices, workers, and resumes.
"""
from __future__ import annotations

import torch

from deepsculpt.core.data.transforms.palette import (
    build_rgba, PaletteConfig, CLASS_PALETTE, _hex_to_rgb,
)


def _sample():
    colors = torch.zeros(2, 4, 4, 4, dtype=torch.int8)
    colors[0, 0, 0, :] = 1   # COL
    colors[0, 0, 1, :] = 4   # PIPE_R
    colors[0, 1, 0, :] = 9   # WALL_R
    colors[1] = colors[0]
    structure = (colors > 0).to(torch.int8)
    index = torch.tensor([3, 42])
    return structure, colors, index


def test_deterministic_across_calls():
    s, c, i = _sample()
    for mode in ("flat", "subtle", "bold"):
        cfg = PaletteConfig(mode=mode, seed=1)
        assert torch.equal(build_rgba(s, c, i, cfg), build_rgba(s, c, i, cfg))


def test_shapes_and_range():
    s, c, i = _sample()
    a = build_rgba(s, c, i, PaletteConfig("subtle", 0))
    assert a.shape == (2, 4, 4, 4, 4)
    assert a.min() >= 0.0 and a.max() <= 1.0


def test_alpha_is_structure():
    s, c, i = _sample()
    a = build_rgba(s, c, i, PaletteConfig("bold", 0))
    assert torch.equal(a[:, 0], s.float())


def test_flat_is_exact_palette():
    s, c, i = _sample()
    a = build_rgba(s, c, i, PaletteConfig("flat"))
    got = a[0, 1:, 0, 1, 0]                       # PIPE_R voxel, rgb channels
    exp = torch.tensor(_hex_to_rgb(CLASS_PALETTE[4]))
    assert torch.allclose(got, exp, atol=1e-4)


def test_empty_voxels_neutral():
    s, c, i = _sample()
    a = build_rgba(s, c, i, PaletteConfig("subtle", 0))
    assert torch.allclose(a[0, 1:, 1, 1, 0], torch.full((3,), 0.5))


def test_shared_hex_diverges_per_sample():
    # PIPE_R(4) and WALL_R(9) share a base hex but get independent per-slot jitter
    s, c, i = _sample()
    a = build_rgba(s, c, i, PaletteConfig("subtle", 0))
    assert not torch.allclose(a[0, 1:, 0, 1, 0], a[0, 1:, 1, 0, 0], atol=1e-3)


def test_per_sample_variety():
    s, c, i = _sample()
    a = build_rgba(s, c, i, PaletteConfig("subtle", 0))
    assert not torch.allclose(a[0, 1:, 0, 1, 0], a[1, 1:, 0, 1, 0], atol=1e-3)


def test_draw_layout_index_stable_under_subset():
    # index, not batch position, determines colour: sample 42 identical whether
    # it appears first or second in the batch.
    s, c, i = _sample()
    a = build_rgba(s, c, i, PaletteConfig("bold", 0))
    s2, c2 = s.flip(0), c.flip(0)
    i2 = i.flip(0)
    b = build_rgba(s2, c2, i2, PaletteConfig("bold", 0))
    assert torch.allclose(a[1], b[0])           # sample idx 42 in both


def test_global_rng_untouched():
    s, c, i = _sample()
    torch.manual_seed(123)
    before = torch.rand(1).item()
    torch.manual_seed(123)
    build_rgba(s, c, i, PaletteConfig("bold", 7))
    after = torch.rand(1).item()
    assert before == after
