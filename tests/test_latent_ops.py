"""Pure tensor tests for deepsculpt.core.latent.ops — no model or data needed.

(The legacy unit/integration suites import the dead `deepSculpt` casing and
don't run; this file is importable and is the smoke gate for latent ops.)
"""

from __future__ import annotations

import pytest
import torch

from deepsculpt.core.latent.ops import (
    latent_arithmetic,
    lerp,
    seeded_z,
    slerp,
    traverse_dimension,
    walk_path,
)


def test_seeded_z_reproducible():
    assert torch.equal(seeded_z(42, 16), seeded_z(42, 16))
    assert not torch.equal(seeded_z(42, 16), seeded_z(43, 16))


def test_lerp_endpoints_and_shape():
    z1, z2 = seeded_z(0, 8)[0], seeded_z(1, 8)[0]
    path = lerp(z1, z2, 5)
    assert path.shape == (5, 8)
    assert torch.allclose(path[0], z1)
    assert torch.allclose(path[-1], z2)


def test_slerp_endpoints_and_norm_preservation():
    z1, z2 = seeded_z(0, 64)[0], seeded_z(1, 64)[0]
    n = z1.norm()
    z2 = z2 / z2.norm() * n  # equal norms -> slerp keeps them along the path
    path = slerp(z1, z2, 7)
    assert path.shape == (7, 64)
    assert torch.allclose(path[0], z1, atol=1e-5)
    assert torch.allclose(path[-1], z2, atol=1e-5)
    norms = path.norm(dim=1)
    assert torch.allclose(norms, n.expand(7), rtol=1e-3)


def test_slerp_parallel_falls_back_to_lerp():
    z = seeded_z(0, 8)[0]
    path = slerp(z, z.clone(), 3)
    assert torch.allclose(path[1], z, atol=1e-5)


def test_slerp_works_on_noise_tensors():
    n1, n2 = torch.randn(1, 1, 4, 4, 4), torch.randn(1, 1, 4, 4, 4)
    path = slerp(n1, n2, 4)
    assert path.shape == (4, 1, 1, 4, 4, 4)


def test_walk_path_no_duplicate_joints():
    anchors = torch.stack([seeded_z(s, 8)[0] for s in (0, 1, 2)])
    path = walk_path(anchors, 5, mode="lerp")
    assert path.shape == (9, 8)  # 5 + (5-1)
    closed = walk_path(anchors, 5, mode="lerp", closed=True)
    assert closed.shape == (13, 8)
    assert torch.allclose(closed[-1], anchors[0], atol=1e-5)


def test_traverse_dimension():
    z = seeded_z(0, 8)[0]
    out = traverse_dimension(z, 3, 5, sigma_range=2.0)
    assert out.shape == (5, 8)
    assert torch.allclose(out[:, 3], torch.linspace(-2.0, 2.0, 5))
    mask = torch.ones(8, dtype=torch.bool)
    mask[3] = False
    assert torch.equal(out[:, mask], z[mask].expand(5, 7))


def test_latent_arithmetic():
    zs = {"a": torch.ones(4), "b": torch.full((4,), 2.0), "c": torch.full((4,), 3.0)}
    out = latent_arithmetic(zs, "a - b + c")
    assert torch.allclose(out, torch.full((4,), 2.0))
    with pytest.raises(KeyError):
        latent_arithmetic(zs, "a + d")
    with pytest.raises(ValueError):
        latent_arithmetic(zs, "a * b")
