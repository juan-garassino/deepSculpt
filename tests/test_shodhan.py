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


def test_slabs_full_square_and_always_ground():
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


def test_strips_only_intermediate_and_rim_preserved():
    for seed in range(30):
        rng = np.random.default_rng(seed)
        sk = sh.build_skeleton(rng)
        strips = sh.cut_slab_strips(sk, rng)
        x0, x1, y0, y1 = sk.plot
        assert sk.slabs[0] not in strips and sk.slabs[-1] not in strips
        for z, (axis, a, b) in strips.items():
            plane = sk.volume[:, :, z]
            # rim ring intact on the cut slab
            assert (plane[x0, y0:y1 + 1] != 0).all() and (plane[x1, y0:y1 + 1] != 0).all()
            assert (plane[x0:x1 + 1, y0] != 0).all() and (plane[x0:x1 + 1, y1] != 0).all()
            # something actually removed strictly between two column rows
            if axis == 0:
                assert (plane[a:b, y0 + 1:y1] == 0).any()
            else:
                assert (plane[x0 + 1:x1, a:b] == 0).any()


def test_l_borders_all_sides_up_down():
    for seed in range(10):
        rng = np.random.default_rng(seed)
        sk = sh.build_skeleton(rng)
        sh.add_l_borders(sk)
        x0, x1, y0, y1 = sk.plot
        for z in sk.slabs:
            for zz in (z + sh.SLAB_T, z - 1):
                if zz < 0 or zz >= sh.N:
                    continue
                band = sk.volume[x0:x1 + 1, y0, zz]
                assert (band != 0).all(), f"seed {seed}: L-band gap at z={zz}"
