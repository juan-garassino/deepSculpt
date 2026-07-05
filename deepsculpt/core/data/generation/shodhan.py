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
            v[cx:cx + COL_T, cy:cy + COL_T, 0:slabs[-1] + SLAB_T] = COL
    for z in slabs:
        put(v, np.s_[x0:x1 + 1, y0:y1 + 1, z:z + SLAB_T], SLAB)

    params = {
        "grammar_version": GRAMMAR_VERSION,
        "storey_height": hz,
        "n_intermediate": n_intermediate,
        "array": [nx, ny],
        "cols_x": cols_x,
        "cols_y": cols_y,
        "terrace": bool(terrace),
    }
    return Skeleton(v, cols_x, cols_y, slabs, (x0, x1, y0, y1), terrace, roof_plane, params)
