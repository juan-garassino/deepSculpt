"""Shodhan grammar — deterministic 64^3 architectural dataset generator.

Spec: docs/superpowers/specs/2026-07-05-shodhan-dataset-design.md
Reference mockups (32^3, user-validated): .superpowers/brainstorm/mockups_v4.py

Every public entry point takes a numpy Generator; generation is a pure
function of (GRAMMAR_VERSION, seed).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

GRAMMAR_VERSION = "2.0.0"

N = 64
MARGIN = 4                      # plot square [MARGIN .. N-1-MARGIN]^2 — constant
SLAB_T = 2                      # slabs are the one thick element
GAP = 6                         # walls/blocks keep this clearance to the plot edge
COL_T = 2                       # column cross-section (2x2)  # tunable at visual gate

# element classes (written into the colors tensor)
COL, SLAB, SCREEN, PIPE_R, PIPE_B, PIPE_Y, VOL, EDGE = 1, 2, 3, 4, 5, 6, 7, 8
WALL_R, WALL_B, WALL_Y, WALL_G = 9, 10, 11, 12
WALL_KINDS = (WALL_R, WALL_B, WALL_Y, WALL_G)
PIPE_KINDS = (PIPE_R, PIPE_B, PIPE_Y)

STOREY_HEIGHTS = (14, 16, 20, 26)
STOREY_WEIGHTS = (0.35, 0.35, 0.2, 0.1)


def put(v: np.ndarray, sl, kind: int) -> None:
    """Fill only empty voxels — elements never overwrite each other."""
    reg = v[sl]
    v[sl] = np.where(reg == 0, kind, reg)


@dataclass
class Skeleton:
    volume: np.ndarray
    cols_x: List[int]
    cols_y: List[int]
    slabs: List[int]            # z of every EXISTING slab (terrace: roof omitted)
    plot: Tuple[int, int, int, int]
    terrace: bool
    roof_plane: int             # z where the roof slab is / would be
    params: Dict = field(default_factory=dict)


def _structured_lines(rng: np.random.Generator, lo: int, hi: int, n: int) -> List[int]:
    """Evenly spaced column lines, inset 4-8 per side, spacing exactly uniform."""
    candidates = []
    for il in range(4, 9):
        for ir in range(4, 9):
            span = (hi - lo) - il - ir
            if span >= n - 1 and span % (n - 1) == 0:
                candidates.append((abs(il - ir), il, ir))
    if candidates:
        best = min(c[0] for c in candidates)
        pool = [c for c in candidates if c[0] == best]
        _, il, ir = pool[int(rng.integers(0, len(pool)))]
        step = ((hi - lo) - il - ir) // (n - 1)
        return [lo + il + k * step for k in range(n)]
    i = int(rng.integers(4, 9))
    step = ((hi - lo) - 2 * i) // (n - 1)
    start = lo + ((hi - lo) - step * (n - 1)) // 2
    return [start + k * step for k in range(n)]


def build_skeleton(rng: np.random.Generator) -> Skeleton:
    v = np.zeros((N, N, N), dtype=np.int8)
    x0, x1 = MARGIN, N - 1 - MARGIN
    y0, y1 = MARGIN, N - 1 - MARGIN

    hz = int(rng.choice(STOREY_HEIGHTS, p=STOREY_WEIGHTS))
    slab_zs = list(range(0, N - SLAB_T - 1, hz))
    n_intermediate = max(0, len(slab_zs) - 2)

    nx, ny = int(rng.choice([3, 4])), int(rng.choice([3, 4]))
    cols_x = _structured_lines(rng, x0, x1, nx)
    cols_y = _structured_lines(rng, y0, y1, ny)

    # terrace: only 3-intermediate buildings may omit the roof slab (p=0.45)
    terrace = n_intermediate >= 3 and rng.random() < 0.45
    roof_plane = slab_zs[-1]
    slabs = slab_zs[:-1] if terrace else slab_zs

    for cx in cols_x:
        for cy in cols_y:
            # Direct assignment (not put()) — COL is the priority element; all
            # other elements use put() and will not overwrite columns.
            v[cx:cx + COL_T, cy:cy + COL_T, 0:slabs[-1] + SLAB_T] = COL
    for z in slabs:
        put(v, np.s_[x0:x1 + 1, y0:y1 + 1, z:z + SLAB_T], SLAB)

    params = {
        "grammar_version": GRAMMAR_VERSION,
        "storey_height": hz,
        "n_intermediate": n_intermediate,
        "array": [nx, ny],
        "cols_x": list(cols_x),
        "cols_y": list(cols_y),
        "terrace": bool(terrace),
    }
    return Skeleton(v, cols_x, cols_y, slabs, (x0, x1, y0, y1), terrace, roof_plane, params)


def cut_slab_strips(sk: Skeleton, rng: np.random.Generator) -> Dict[int, Tuple[int, int, int]]:
    """One strip per INTERMEDIATE slab, between two adjacent column rows,
    rim ring preserved. Returns {slab_z: (axis, a, b)} for exclusion rules."""
    v = sk.volume
    x0, x1, y0, y1 = sk.plot
    strips: Dict[int, Tuple[int, int, int]] = {}
    for z in sk.slabs[1:-1]:
        axis = int(rng.integers(0, 2))
        lines = sorted(sk.cols_x if axis == 0 else sk.cols_y)
        if len(lines) < 2:
            continue
        i = int(rng.integers(0, len(lines) - 1))
        a, b = lines[i] + COL_T, lines[i + 1]
        if b <= a:
            continue
        for dz in range(SLAB_T):
            if axis == 0:
                reg = v[a:b, y0 + 1:y1, z + dz]
            else:
                reg = v[x0 + 1:x1, a:b, z + dz]
            reg[reg == SLAB] = 0
        strips[z] = (axis, a, b)
    sk.params["strips"] = {int(z): [int(axis), int(a), int(b)] for z, (axis, a, b) in strips.items()}
    return strips


def add_l_borders(sk: Skeleton) -> None:
    """White band on every slab rim: all four sides, above AND below."""
    v = sk.volume
    x0, x1, y0, y1 = sk.plot
    for z in sk.slabs:
        for zz in (z + SLAB_T, z - 1):
            if zz < 0 or zz >= N:
                continue
            put(v, np.s_[x0:x1 + 1, y0:y0 + 1, zz:zz + 1], EDGE)
            put(v, np.s_[x0:x1 + 1, y1:y1 + 1, zz:zz + 1], EDGE)
            put(v, np.s_[x0:x0 + 1, y0:y1 + 1, zz:zz + 1], EDGE)
            put(v, np.s_[x1:x1 + 1, y0:y1 + 1, zz:zz + 1], EDGE)


@dataclass
class Wall:
    axis: int
    pos: int
    s0: int
    s1: int
    z_lo: int
    z_hi: int
    level: int
    floor_z: int
    double: bool
    kind: int


def _wall_hits_strip(axis: int, pos: int, s0: int, s1: int, strip) -> bool:
    st_axis, a, b = strip
    if st_axis == axis:
        return a <= pos < b
    return not (s1 < a or s0 >= b)


def _point_in_strip(x: int, y: int, strip) -> bool:
    st_axis, a, b = strip
    return (a <= x < b) if st_axis == 0 else (a <= y < b)


def add_pipes(sk: Skeleton, rng: np.random.Generator, walls: List[Wall], strips):
    """2-3 pipes: 1x1 riser hugging a wall face ground->served slab, plus a
    horizontal run under that slab, clipped at strips, strictly interior."""
    v = sk.volume
    x0, x1, y0, y1 = sk.plot
    lo, hi = x0 + 1, x1 - 1
    kinds = list(PIPE_KINDS)
    rng.shuffle(kinds)
    target = int(rng.integers(2, 4))
    placed = []
    for _ in range(60):
        if len(placed) >= target or not walls:
            break
        w = walls[int(rng.integers(0, len(walls)))]
        k = kinds[len(placed) % 3]
        off = int(rng.choice([-1, 1]))
        span = w.s1 - w.s0
        along = int(rng.integers(w.s0, max(w.s0 + 1, w.s1)))
        rx = w.pos + off if w.axis == 0 else along
        ry = along if w.axis == 0 else w.pos + off
        rx, ry = int(np.clip(rx, lo, hi)), int(np.clip(ry, lo, hi))
        z_top = min(w.z_hi + 1, N - 1)
        if any(_point_in_strip(rx, ry, st) for z, st in strips.items() if 0 < z <= z_top):
            continue
        v[rx, ry, 0:z_top][v[rx, ry, 0:z_top] == 0] = k
        # horizontal run under the slab above, clipped at a strip
        run = int(span * rng.uniform(0.5, 1.0))
        zh = max(0, w.z_hi)
        strip_above = strips.get(w.z_hi + 1)
        if w.axis == 0:
            a, b = sorted((ry, int(np.clip(ry + rng.choice([-1, 1]) * run, lo, hi))))
            if strip_above and strip_above[0] == 1:
                sa, sb = strip_above[1], strip_above[2]
                if a < sa <= b:
                    b = sa - 1
                elif a <= sb - 1 < b:
                    a = sb
            if b >= a:
                put(v, np.s_[rx, a:b + 1, zh:zh + 1], k)
        else:
            a, b = sorted((rx, int(np.clip(rx + rng.choice([-1, 1]) * run, lo, hi))))
            if strip_above and strip_above[0] == 0:
                sa, sb = strip_above[1], strip_above[2]
                if a < sa <= b:
                    b = sa - 1
                elif a <= sb - 1 < b:
                    a = sb
            if b >= a:
                put(v, np.s_[a:b + 1, ry, zh:zh + 1], k)
        placed.append((rx, ry, z_top, k))
    sk.params["n_pipes"] = len(placed)
    return placed


def add_walls(sk: Skeleton, rng: np.random.Generator, strips) -> List[Wall]:
    """Two walls per level: on grid lines, endpoints at columns, >=2 bays
    when the array allows, GAP-clear of edges, never standing on strips;
    ~30% of levels get a double-height second wall."""
    v = sk.volume
    x0, x1, y0, y1 = sk.plot
    walls: List[Wall] = []
    for gi in range(len(sk.slabs) - 1):
        z_lo = sk.slabs[gi] + SLAB_T
        floor_strip = strips.get(sk.slabs[gi])
        ceil_strip = strips.get(sk.slabs[gi + 1])
        placed = 0
        for _ in range(48):
            if placed >= 2:
                break
            double = (placed == 1 and gi + 2 <= len(sk.slabs) - 1 and rng.random() < 0.3)
            z_hi = sk.slabs[gi + 2] - 1 if double else sk.slabs[gi + 1] - 1
            axis = int(rng.integers(0, 2))
            on_lines = sk.cols_x if axis == 0 else sk.cols_y
            across = sk.cols_y if axis == 0 else sk.cols_x
            on_ok = [c for c in on_lines if x0 + GAP <= c <= x1 - GAP]
            if not on_ok or len(across) < 2:
                continue
            pos = int(rng.choice(on_ok))
            i = int(rng.integers(0, len(across) - 1))
            j_max = min(i + 2, len(across) - 1)
            j = int(rng.integers(i + 1, j_max + 1))
            s0, s1 = across[i], across[j]
            if floor_strip and _wall_hits_strip(axis, pos, s0, s1, floor_strip):
                continue
            if not double and ceil_strip and _wall_hits_strip(axis, pos, s0, s1, ceil_strip):
                continue
            kind = int(rng.choice(WALL_KINDS))
            if axis == 0:
                put(v, np.s_[pos, s0:s1 + 1, z_lo:z_hi + 1], kind)
            else:
                put(v, np.s_[s0:s1 + 1, pos, z_lo:z_hi + 1], kind)
            walls.append(Wall(axis, pos, s0, s1, z_lo, z_hi, gi, sk.slabs[gi], double, kind))
            placed += 1
    sk.params["n_walls"] = len(walls)
    sk.params["n_double_walls"] = sum(1 for w in walls if w.double)
    return walls


def add_screens(sk: Skeleton, rng: np.random.Generator) -> Dict:
    """Brise-soleil: 1-voxel bars, 2 deep, flush at the slab rim plane,
    ground -> last slab. Vertical fins only, EXCEPT single-intermediate
    buildings which may use the full lattice (p=0.7). Present p=0.75."""
    info = {"sides": [], "lattice": False}
    if rng.random() >= 0.75:
        sk.params["screens"] = info
        return info
    v = sk.volume
    x0, x1, y0, y1 = sk.plot
    n = int(rng.integers(1, 3))
    sides = [int(s) for s in rng.permutation(4)[:n]]
    rhythm = int(rng.integers(6, 9))
    lattice = sk.params["n_intermediate"] <= 1 and rng.random() < 0.7
    for side in sides:
        if side == 0:
            depth = (x0, x0 + 1)
        elif side == 1:
            depth = (x1, x1 - 1)
        elif side == 2:
            depth = (y0, y0 + 1)
        else:
            depth = (y1, y1 - 1)
        rng_a = range(y0, y1 + 1) if side < 2 else range(x0, x1 + 1)
        for a in rng_a:
            for zz in range(SLAB_T, sk.slabs[-1]):
                if lattice:
                    if a % rhythm != 0 and zz % rhythm != 0:
                        continue
                elif a % rhythm != 0:
                    continue
                for dpos in depth:
                    p = (dpos, a) if side < 2 else (a, dpos)
                    if v[p[0], p[1], zz] == 0:
                        v[p[0], p[1], zz] = SCREEN
    info.update({"sides": sides, "lattice": bool(lattice), "rhythm": rhythm})
    sk.params["screens"] = info
    return info
