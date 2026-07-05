"""Tests for the Shodhan grammar generator (spec 2026-07-05).

Pure numpy — no torch needed for grammar logic."""
import numpy as np
import pytest

from deepsculpt.core.data.generation import shodhan as sh


def test_constants():
    assert sh.N == 64
    assert sh.MARGIN == 4
    assert sh.GAP == 6


def test_skeleton_deterministic():
    a = sh.build_skeleton(np.random.default_rng(7))
    b = sh.build_skeleton(np.random.default_rng(7))
    assert np.array_equal(a.volume, b.volume)
    assert a.params == b.params


def test_column_array_structured():
    for seed in range(20):
        sk = sh.build_skeleton(np.random.default_rng(seed))
        # 3x3 / 3x4 / 4x4 arrays
        assert len(sk.cols_x) in (3, 4) and len(sk.cols_y) in (3, 4)
        # evenly spaced
        dx = np.diff(sk.cols_x)
        assert (dx == dx[0]).all(), f"seed {seed}: uneven column spacing {dx}"
        # inset 4-8 from plot edge
        assert sk.cols_x[0] - sh.MARGIN >= 4 and (sh.N - 1 - sh.MARGIN) - sk.cols_x[-1] >= 4


def test_slabs_full_square_and_always_ground_roof():
    for seed in range(20):
        sk = sh.build_skeleton(np.random.default_rng(seed))
        x0, x1, y0, y1 = sk.plot
        for z in sk.slabs:
            plane = sk.volume[x0:x1 + 1, y0:y1 + 1, z:z + sh.SLAB_T]
            assert (plane != 0).all(), f"seed {seed}: slab at z={z} not full"
        assert sk.slabs[0] == 0


def test_terrace_only_on_three_intermediate():
    saw_terrace = False
    for seed in range(200):
        sk = sh.build_skeleton(np.random.default_rng(seed))
        n_int = len(sk.slabs) - 2 + (1 if sk.terrace else 0)  # terrace removed the roof slab
        if sk.terrace:
            saw_terrace = True
            assert n_int == 3, f"seed {seed}: terrace on {n_int}-intermediate building"
            # no columns above the last slab
            above = sk.volume[:, :, sk.slabs[-1] + sh.SLAB_T:] == sh.COL
            assert not above.any()
    assert saw_terrace
