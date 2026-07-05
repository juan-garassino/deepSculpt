"""Tests for the Shodhan grammar generator (spec 2026-07-05).

Pure numpy — no torch needed for grammar logic."""
import numpy as np
import pytest

from deepsculpt.core.data.generation import shodhan as sh

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


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
                for band in (
                    sk.volume[x0:x1 + 1, y0, zz],
                    sk.volume[x0:x1 + 1, y1, zz],
                    sk.volume[x0, y0:y1 + 1, zz],
                    sk.volume[x1, y0:y1 + 1, zz],
                ):
                    assert (band != 0).all(), f"seed {seed}: L-band gap at z={zz}"


def test_walls_rules():
    for seed in range(30):
        rng = np.random.default_rng(seed)
        sk = sh.build_skeleton(rng)
        strips = sh.cut_slab_strips(sk, rng)
        walls = sh.add_walls(sk, rng, strips)
        x0, x1, y0, y1 = sk.plot
        # exactly two per level
        per_level = {}
        for w in walls:
            per_level[w.level] = per_level.get(w.level, 0) + 1
        assert all(c == 2 for c in per_level.values()), f"seed {seed}: {per_level}"
        assert len(per_level) == len(sk.slabs) - 1, f"seed {seed}: missing levels"
        for w in walls:
            # on a column line, endpoints at columns
            lines = sk.cols_x if w.axis == 0 else sk.cols_y
            across = sk.cols_y if w.axis == 0 else sk.cols_x
            assert w.pos in lines
            assert w.s0 in across and w.s1 in across
            # edge clearance for the wall's own line
            assert x0 + sh.GAP <= w.pos <= x1 - sh.GAP
            # never standing on a strip
            floor_strip = strips.get(w.floor_z)
            if floor_strip:
                assert not sh._wall_hits_strip(w.axis, w.pos, w.s0, w.s1, floor_strip)
            if not w.double:
                ceil_strip = strips.get(sk.slabs[w.level + 1])
                if ceil_strip:
                    assert not sh._wall_hits_strip(w.axis, w.pos, w.s0, w.s1, ceil_strip)


def test_pipes_rules():
    for seed in range(30):
        rng = np.random.default_rng(seed)
        sk = sh.build_skeleton(rng)
        strips = sh.cut_slab_strips(sk, rng)
        walls = sh.add_walls(sk, rng, strips)
        pipes = sh.add_pipes(sk, rng, walls, strips)
        assert 2 <= len(pipes) <= 3, f"seed {seed}: {len(pipes)} pipes"
        x0, x1, y0, y1 = sk.plot
        for rx, ry, z_top, k in pipes:
            # strictly inside the facade plane
            assert x0 < rx < x1 and y0 < ry < y1
            # riser never inside a strip void it crosses
            for z, st in strips.items():
                if 0 < z <= z_top:
                    assert not sh._point_in_strip(rx, ry, st)


def test_screens_flush_and_fins():
    lattice_seen = fins_seen = False
    for seed in range(60):
        rng = np.random.default_rng(seed)
        sk = sh.build_skeleton(rng)
        info = sh.add_screens(sk, rng)
        if not info["sides"]:
            continue
        x0, x1, y0, y1 = sk.plot
        scr = sk.volume == sh.SCREEN
        assert scr.any()
        # flush: screen voxels only on rim planes or one voxel inward
        xs, ys, zs = np.where(scr)
        ok = ((xs == x0) | (xs == x0 + 1) | (xs == x1) | (xs == x1 - 1)
              | (ys == y0) | (ys == y0 + 1) | (ys == y1) | (ys == y1 - 1))
        assert ok.all(), f"seed {seed}: screen voxel off the rim planes"
        # vertical span: ground slab top to below last slab
        assert zs.min() >= sh.SLAB_T and zs.max() < sk.slabs[-1]
        if info["lattice"]:
            lattice_seen = True
            assert sk.params["n_intermediate"] <= 1
        else:
            fins_seen = True
    assert fins_seen


def test_massing_carved_and_thin_shells():
    for seed in range(30):
        rng = np.random.default_rng(seed)
        sk = sh.build_skeleton(rng)
        blocks = sh.add_massing(sk, rng, mass=0.8)
        assert blocks, "dense mass dial must produce blocks"
        for (bx, by, w, d, z_lo, h) in blocks:
            solid = (sk.volume[bx:bx + w, by:by + d, z_lo:z_lo + h] == sh.VOL).mean()
            assert solid <= 0.7 + 1e-6, f"seed {seed}: block {solid:.2f} solid"


def test_terrace_composition():
    found = 0
    for seed in range(300):
        rng = np.random.default_rng(seed)
        sk = sh.build_skeleton(rng)
        if not sk.terrace:
            continue
        sh.add_terrace(sk, rng)
        found += 1
        z_lo = sk.slabs[-1] + sh.SLAB_T
        top = sk.volume[:, :, z_lo:sk.roof_plane + 1]
        # no columns on the terrace
        assert not (top == sh.COL).any()
        # something reaches the roof plane (envelope reads full height)
        assert (sk.volume[:, :, sk.roof_plane] != 0).any()
        # exactly 2 terrace walls recorded
        assert sk.params["terrace_info"]["n_walls"] == 2
        if found >= 5:
            break
    assert found >= 1


def test_generate_structure_end_to_end():
    a, pa = sh.generate_structure_with_params(7)
    b, pb = sh.generate_structure_with_params(7)
    assert np.array_equal(a, b) and pa == pb          # determinism
    for seed in range(20):
        v, p = sh.generate_structure_with_params(seed)
        assert v.shape == (64, 64, 64)
        occ = (v > 0).mean()
        assert 0.05 <= occ <= 0.35, f"seed {seed}: occupancy {occ:.3f}"
        # invariants: elements present
        assert (v == sh.COL).any() and (v == sh.SLAB).any()
        assert np.isin(v, sh.WALL_KINDS).any()
        assert np.isin(v, sh.PIPE_KINDS).any()
        # single connected component
        from scipy.ndimage import label
        _, n = label(v > 0)
        assert n == 1, f"seed {seed}: {n} components"
        # nothing outside the plot square
        assert (v[:4, :, :] == 0).all() and (v[60:, :, :] == 0).all()
        assert (v[:, :4, :] == 0).all() and (v[:, 60:, :] == 0).all()


def test_write_dataset(tmp_path):
    import json, torch
    out = sh.write_dataset(tmp_path, num_samples=4, seed_start=100)
    st = sorted((out / "pytorch_samples" / "structures").glob("structure_*.pt"))
    co = sorted((out / "pytorch_samples" / "colors").glob("colors_*.pt"))
    pr = sorted((out / "params").glob("params_*.json"))
    assert len(st) == len(co) == len(pr) == 4
    s = torch.load(st[0], map_location="cpu")
    c = torch.load(co[0], map_location="cpu")
    assert s.shape == (64, 64, 64) and s.dtype == torch.int8
    assert set(s.unique().tolist()) <= {0, 1}
    assert c.shape == (64, 64, 64)                    # element classes
    meta = json.loads((out / "dataset_metadata.json").read_text())
    assert "occupancy_stats" in meta and "variant_distribution" in meta
    assert meta["grammar_version"] == sh.GRAMMAR_VERSION


def test_probe_gates_smoke(tmp_path):
    from scripts.shodhan_probe import probe_gates
    report = probe_gates(n=8)
    assert set(report) >= {"pairwise_iou_masked", "max_freq_outside_template", "pass"}
    assert 0.0 <= report["pairwise_iou_masked"] <= 1.0
