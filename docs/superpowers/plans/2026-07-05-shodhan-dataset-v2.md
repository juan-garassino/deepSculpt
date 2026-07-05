# Shodhan Dataset v2 + Baseline Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the user-approved Shodhan grammar (spec: `docs/superpowers/specs/2026-07-05-shodhan-dataset-design.md`) as a deterministic 64³ dataset generator with quality gates and a visual approval gate, then run baseline GAN + diffusion training on the new data using the proven Cloud Run recipes.

**Architecture:** A pure-function generator module (`shodhan.py`, port of the validated mockup grammar `.superpowers/brainstorm/mockups_v4.py` scaled ×2) returns `(structure, colors, params)` per seed. A writer lays out samples in the existing loader format plus `params_*.json`. A quality module implements per-sample and dataset-level gates; a probe script renders contact sheets + 3D HTML for the user's visual gate. CLI gains `--structure-preset shodhan` (new default). Training reuses existing trainers/infra unmodified.

**Tech Stack:** Python 3.10+, numpy, torch (save/load only in writer), scipy (connectivity), matplotlib + plotly (probe renders), pytest. No GPU needed until the training phase.

**Reference:** the mockup module `.superpowers/brainstorm/mockups_v4.py` is the validated grammar at 32³. Production code is the same logic at 64³ with the scale map below. Where this plan's code differs from the mockup (params capture, integrity checks, no stairs), this plan wins.

**Scale map (mockup 32³ → production 64³):** margin 2→4 (plot [4..59]²), column inset 2–4→4–8, GAP 3→6, storey heights {7,8,10,13}→{14,16,20,26} (weights 0.35/0.35/0.2/0.1), screen rhythm 3–4→6–8, block footprints 8–16→16–32, wall spans ×2, slab thickness 1→2, column 1×1→2×2 ⚙, thin members (walls, fins, pipes, shells, L-band) stay **1 voxel** (user rule), fins depth 2.

---

### Task 0: Branch + dependency

**Files:**
- Modify: `pyproject.toml` (add scipy to dependencies)

- [ ] **Step 1: Confirm branch**

```bash
git checkout feat/shodhan-dataset  # already exists, spec committed here
git log --oneline -1               # expect: docs(spec) commit
```

- [ ] **Step 2: Add scipy dependency**

In `pyproject.toml`, find the `[project]` `dependencies` list and add `"scipy>=1.10"` (used for `scipy.ndimage.label` connectivity checks). If scipy is already present, skip.

- [ ] **Step 3: Sync and verify import**

```bash
uv sync --extra dev 2>/dev/null || pip install scipy
python -c "import scipy.ndimage; print('ok')"
```
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add scipy for dataset connectivity gates"
```

---

### Task 1: Grammar module — constants, skeleton (columns, slabs, terrace flag)

**Files:**
- Create: `deepsculpt/core/data/generation/shodhan.py`
- Test: `tests/test_shodhan.py`

- [ ] **Step 1: Write failing tests for the skeleton**

Create `tests/test_shodhan.py`:

```python
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
        assert sh.MARGIN + 4 <= sk.cols_x[0] and sk.cols_x[-1] <= sh.N - 1 - sh.MARGIN - 4 or True
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
```

- [ ] **Step 2: Run tests, expect import failure**

```bash
uv run pytest tests/test_shodhan.py -x -q 2>&1 | tail -3
```
Expected: `ModuleNotFoundError` / `ImportError` on `shodhan`.

- [ ] **Step 3: Implement constants + skeleton**

Create `deepsculpt/core/data/generation/shodhan.py`:

```python
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
    return sorted({int(round(p)) for p in np.linspace(lo + i, hi - i, n)})


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
```

Also create/append the package export in `deepsculpt/core/data/generation/__init__.py` only if other modules are exported there by name (check first; if it uses explicit imports, add nothing — tests import `... generation import shodhan` directly, which works without `__init__` changes).

- [ ] **Step 4: Run tests, expect pass**

```bash
uv run pytest tests/test_shodhan.py -x -q 2>&1 | tail -3
```
Expected: all pass. (If `test_column_array_structured` inset assertion trips, the bug is in `_structured_lines` inset range — fix there, not in the test.)

- [ ] **Step 5: Commit**

```bash
git add deepsculpt/core/data/generation/shodhan.py tests/test_shodhan.py
git commit -m "feat(shodhan): grammar skeleton — structured column arrays, full-square slabs, terrace flag"
```

---

### Task 2: Strips + L-borders

**Files:**
- Modify: `deepsculpt/core/data/generation/shodhan.py`
- Test: `tests/test_shodhan.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_shodhan.py`:

```python
def _full_build(seed):
    return sh.generate_structure(np.random.default_rng(seed))


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


def test_l_borders_all_sides_up_down():
    for seed in range(10):
        rng = np.random.default_rng(seed)
        sk = sh.build_skeleton(rng)
        sh.add_l_borders(sk)
        x0, x1, y0, y1 = sk.plot
        for z in sk.slabs:
            for dz in (SLAB_UP := z + sh.SLAB_T, z - 1):
                if dz < 0 or dz >= sh.N:
                    continue
                band = sk.volume[x0:x1 + 1, y0, dz]
                assert (band == sh.EDGE).all() or (band != 0).all()
```

- [ ] **Step 2: Run, expect AttributeError (functions missing)**

```bash
uv run pytest tests/test_shodhan.py -x -q 2>&1 | tail -3
```

- [ ] **Step 3: Implement**

Append to `shodhan.py`:

```python
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
```

- [ ] **Step 4: Run tests, expect pass**

```bash
uv run pytest tests/test_shodhan.py -x -q 2>&1 | tail -3
```

- [ ] **Step 5: Commit**

```bash
git add deepsculpt/core/data/generation/shodhan.py tests/test_shodhan.py
git commit -m "feat(shodhan): slab strips (rim preserved) + L-borders"
```

---

### Task 3: Walls (column-to-column, strip-safe, double-height)

**Files:**
- Modify: `deepsculpt/core/data/generation/shodhan.py`
- Test: `tests/test_shodhan.py`

- [ ] **Step 1: Write failing tests**

Append:

```python
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
        for w in walls:
            # on a column line, endpoints at columns, >= 2 bays when possible
            lines = sk.cols_x if w.axis == 0 else sk.cols_y
            across = sk.cols_y if w.axis == 0 else sk.cols_x
            assert w.pos in lines
            assert w.s0 in across and w.s1 in across
            # edge clearance
            assert w.pos >= x0 + sh.GAP and w.pos <= x1 - sh.GAP
            # never standing on a strip
            floor_strip = strips.get(w.floor_z)
            if floor_strip:
                assert not sh._wall_hits_strip(w.axis, w.pos, w.s0, w.s1, floor_strip)
```

- [ ] **Step 2: Run, expect failure** (`add_walls` missing)

- [ ] **Step 3: Implement**

Append to `shodhan.py`:

```python
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


def add_walls(sk: Skeleton, rng: np.random.Generator, strips) -> List[Wall]:
    """Two walls per level: on grid lines, endpoints at columns, >=2 bays,
    GAP-clear of edges, never standing on strips; ~30% double-height."""
    v = sk.volume
    x0, x1, y0, y1 = sk.plot
    walls: List[Wall] = []
    for gi in range(len(sk.slabs) - 1):
        z_lo = sk.slabs[gi] + SLAB_T
        floor_strip = strips.get(sk.slabs[gi])
        ceil_strip = strips.get(sk.slabs[gi + 1])
        placed = 0
        for _ in range(24):
            if placed >= 2:
                break
            double = (placed == 1 and gi + 2 <= len(sk.slabs) - 1 and rng.random() < 0.3)
            z_hi = sk.slabs[gi + 2] - 1 if double else sk.slabs[gi + 1] - 1
            axis = int(rng.integers(0, 2))
            on_lines = sk.cols_x if axis == 0 else sk.cols_y
            across = sk.cols_y if axis == 0 else sk.cols_x
            on_ok = [c for c in on_lines if x0 + GAP <= c <= x1 - GAP]
            across_ok = [c for c in across if y0 + GAP <= c <= y1 - GAP or True]
            across_ok = [c for c in across]  # endpoints are columns; clearance comes from insets
            if not on_ok or len(across_ok) < 2:
                continue
            pos = int(rng.choice(on_ok))
            i = int(rng.integers(0, len(across_ok) - 1))
            j = min(i + 2, len(across_ok) - 1)
            if j <= i:
                j = i + 1
            s0, s1 = across_ok[i], across_ok[j]
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
```

Note the `across_ok` line: endpoints must be **columns** (wall runs column-to-column); the GAP clearance applies to `pos` (the wall's own line). Column insets (4–8) already keep endpoints off the edge. Delete the dead first `across_ok` assignment when implementing — keep only `across_ok = [c for c in across]`.

- [ ] **Step 4: Run tests, expect pass.** If a seed fails wall count 2 (very constrained geometry), it indicates the resample budget: raise attempts from 24 to 48 rather than weakening assertions.

- [ ] **Step 5: Commit**

```bash
git add -u && git commit -m "feat(shodhan): walls — column-to-column, strip-safe, double-height"
```

---

### Task 4: Pipes (wall-anchored, strip-safe, nothing-out)

**Files:**
- Modify: `deepsculpt/core/data/generation/shodhan.py`
- Test: `tests/test_shodhan.py`

- [ ] **Step 1: Write failing tests**

```python
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
```

- [ ] **Step 2: Run, expect failure**

- [ ] **Step 3: Implement**

```python
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
    for _ in range(30):
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
```

- [ ] **Step 4: Run tests, expect pass.** If some seed yields < 2 pipes, raise attempts to 60 before relaxing anything.

- [ ] **Step 5: Commit**

```bash
git add -u && git commit -m "feat(shodhan): pipes — wall-anchored, strip-safe, strictly interior"
```

---

### Task 5: Screens (vertical fins; lattice for single-intermediate)

**Files:**
- Modify: `deepsculpt/core/data/generation/shodhan.py`
- Test: `tests/test_shodhan.py`

- [ ] **Step 1: Write failing tests**

```python
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
        # flush: screen voxels only on the rim plane and one voxel inward
        xs = sorted(set(np.where(scr)[0]))
        ok_x = {x0, x0 + 1, x1 - 1, x1}
        ys = sorted(set(np.where(scr)[1]))
        ok_y = {y0, y0 + 1, y1 - 1, y1}
        assert set(xs) <= ok_x or set(ys) <= ok_y
        if info["lattice"]:
            lattice_seen = True
            assert sk.params["n_intermediate"] <= 1
        else:
            fins_seen = True
    assert fins_seen
```

- [ ] **Step 2: Run, expect failure**

- [ ] **Step 3: Implement**

```python
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
```

- [ ] **Step 4: Run tests, expect pass**

- [ ] **Step 5: Commit**

```bash
git add -u && git commit -m "feat(shodhan): brise-soleil — flush vertical fins, lattice on grand floors"
```

---

### Task 6: Massing + carving (full-height openings, 1-voxel shells)

**Files:**
- Modify: `deepsculpt/core/data/generation/shodhan.py`
- Test: `tests/test_shodhan.py`

- [ ] **Step 1: Write failing tests**

```python
def test_massing_carved_and_thin_shells():
    for seed in range(30):
        rng = np.random.default_rng(seed)
        sk = sh.build_skeleton(rng)
        blocks = sh.add_massing(sk, rng, mass=0.8)
        assert blocks, "dense mass dial must produce blocks"
        for (bx, by, w, d, z_lo, h) in blocks:
            solid = (sk.volume[bx:bx + w, by:by + d, z_lo:z_lo + h] == sh.VOL).mean()
            assert solid <= 0.7 + 1e-6, f"seed {seed}: block {solid:.2f} solid"
```

- [ ] **Step 2: Run, expect failure**

- [ ] **Step 3: Implement**

```python
def carve_block(v, rng, bx, by, w, d, z_lo, h):
    """All openings full-height (slab to slab) — abstracted doors.
    Shell is ALWAYS 1 voxel: masses never grow thick walls."""
    z_a, z_b = z_lo + SLAB_T, z_lo + h
    if w >= 16 and d >= 16:
        t = 1
        reg = v[bx + t:bx + w - t, by + t:by + d - t, z_a:z_b]
        reg[reg == VOL] = 0
        dw = int(rng.integers(4, 8))
        dp = int(rng.integers(by + 4, max(by + 5, by + d - 4 - dw)))
        reg = v[bx:bx + t + 1, dp:dp + dw, z_a:z_b]
        reg[reg == VOL] = 0
    if rng.random() < 0.8 and w >= 12 and d >= 12:
        tw = int(rng.integers(6, max(7, w // 2)))
        ty = int(rng.integers(by + 2, max(by + 3, by + d - tw - 2)))
        reg = v[bx:bx + w, ty:ty + tw, z_a:z_b]
        reg[reg == VOL] = 0
    for _ in range(int(rng.integers(1, 3))):
        cw = int(rng.integers(6, max(7, w // 2)))
        cd = int(rng.integers(6, max(7, d // 2)))
        side = int(rng.integers(0, 4))
        if side == 0:
            sl = np.s_[bx:bx + cw, by:by + cd, z_a:z_b]
        elif side == 1:
            sl = np.s_[bx + w - cw:bx + w, by + d - cd:by + d, z_a:z_b]
        elif side == 2:
            sl = np.s_[bx:bx + cw, by + d - cd:by + d, z_a:z_b]
        else:
            sl = np.s_[bx + w - cw:bx + w, by:by + cd, z_a:z_b]
        reg = v[sl]
        reg[reg == VOL] = 0


def add_massing(sk: Skeleton, rng: np.random.Generator, mass: Optional[float] = None):
    v = sk.volume
    x0, x1, y0, y1 = sk.plot
    m = mass if mass is not None else float(rng.uniform(0.25, 0.85))
    blocks = []
    for _ in range(int(1 + m * 3)):
        w = int(np.clip(rng.integers(18, 32) * (0.6 + m), 16, x1 - x0 - 2))
        d = int(np.clip(rng.integers(18, 32) * (0.6 + m), 16, y1 - y0 - 2))
        gi = int(rng.integers(0, max(1, len(sk.slabs) - 1)))
        z_lo = sk.slabs[gi]
        top = sk.slabs[min(gi + int(rng.integers(1, 3)), len(sk.slabs) - 1)]
        h = (top - z_lo) or (sk.slabs[1] - sk.slabs[0])
        bx = int(np.clip(int(rng.choice(sk.cols_x)) - rng.integers(0, 5), x0 + 1, max(x0 + 1, x1 - w)))
        by = int(np.clip(int(rng.choice(sk.cols_y)) - rng.integers(0, 5), y0 + 1, max(y0 + 1, y1 - d)))
        put(v, np.s_[bx:bx + w, by:by + d, z_lo:z_lo + h], VOL)
        blocks.append((bx, by, w, d, z_lo, h))
    for b in blocks:
        carve_block(v, rng, *b)
        bx, by, w, d, z_lo, h = b
        for _ in range(4):
            if (v[bx:bx + w, by:by + d, z_lo:z_lo + h] == VOL).mean() <= 0.7:
                break
            carve_block(v, rng, *b)
    sk.params["mass_dial"] = round(m, 3)
    sk.params["n_blocks"] = len(blocks)
    return blocks
```

- [ ] **Step 4: Run tests, expect pass**

- [ ] **Step 5: Commit**

```bash
git add -u && git commit -m "feat(shodhan): carved massing — slab-to-slab openings, 1-voxel shells"
```

---

### Task 7: Terrace (block + two perpendicular walls + clean riser)

**Files:**
- Modify: `deepsculpt/core/data/generation/shodhan.py`
- Test: `tests/test_shodhan.py`

- [ ] **Step 1: Write failing tests**

```python
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
        # two wall colors or one color twice: exactly 2 wall elements — check
        # via params
        assert sk.params["terrace"]["n_walls"] == 2
        if found >= 5:
            break
    assert found >= 1
```

- [ ] **Step 2: Run, expect failure**

- [ ] **Step 3: Implement**

```python
def add_terrace(sk: Skeleton, rng: np.random.Generator) -> None:
    """Terrace: ONE carved block rising to the roof plane + TWO long
    perpendicular walls (1 thick) + optional clean full-height riser."""
    if not sk.terrace:
        return
    v = sk.volume
    x0, x1, y0, y1 = sk.plot
    z_lo = sk.slabs[-1] + SLAB_T
    z_hi = sk.roof_plane

    w = int(rng.integers(20, 32))
    d = int(rng.integers(20, 32))
    bx = int(np.clip(int(rng.choice(sk.cols_x)), x0 + GAP, max(x0 + GAP, x1 - GAP - w)))
    by = int(np.clip(int(rng.choice(sk.cols_y)), y0 + GAP, max(y0 + GAP, y1 - GAP - d)))
    h = z_hi - sk.slabs[-1]
    put(v, np.s_[bx:bx + w, by:by + d, z_lo:z_hi + 1], VOL)
    carve_block(v, rng, bx, by, w, d, sk.slabs[-1], h)

    axis = int(rng.integers(0, 2))
    first_wall = None
    for i, ax in enumerate((axis, 1 - axis)):
        pos = int(rng.integers(x0 + GAP, x1 - GAP + 1))
        span = int((x1 - x0) * rng.uniform(0.55, 0.85))
        s0 = int(rng.integers(y0 + GAP, max(y0 + GAP + 1, y1 - GAP - span + 1)))
        kind = int(rng.choice(WALL_KINDS))
        if ax == 0:
            put(v, np.s_[pos, s0:s0 + span, z_lo:z_hi + 1], kind)
        else:
            put(v, np.s_[s0:s0 + span, pos, z_lo:z_hi + 1], kind)
        if i == 0:
            first_wall = (ax, pos, s0, span)

    has_pipe = False
    if first_wall and rng.random() < 0.5:
        ax, pos, s0, span = first_wall
        k = int(rng.choice(PIPE_KINDS))
        for _ in range(10):
            off = int(rng.choice([-1, 1]))
            along = int(rng.integers(s0 + span // 4, max(s0 + span // 4 + 1, s0 + (3 * span) // 4)))
            rx = pos + off if ax == 0 else along
            ry = along if ax == 0 else pos + off
            rx = int(np.clip(rx, x0 + 1, x1 - 1))
            ry = int(np.clip(ry, y0 + 1, y1 - 1))
            if (v[rx, ry, z_lo:z_hi + 1] == 0).all():
                v[rx, ry, z_lo:z_hi + 1] = k
                has_pipe = True
                break
    sk.params["terrace"] = {"block": [int(bx), int(by), int(w), int(d)],
                            "n_walls": 2, "pipe": bool(has_pipe)}
```

- [ ] **Step 4: Run tests, expect pass**

- [ ] **Step 5: Commit**

```bash
git add -u && git commit -m "feat(shodhan): terrace — carved block, two perpendicular walls, clean riser"
```

---

### Task 8: Assembly — `generate_structure` with integrity gates

**Files:**
- Modify: `deepsculpt/core/data/generation/shodhan.py`
- Test: `tests/test_shodhan.py`

- [ ] **Step 1: Write failing tests**

```python
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
```

- [ ] **Step 2: Run, expect failure**

- [ ] **Step 3: Implement**

```python
def _integrity_ok(v: np.ndarray) -> Tuple[bool, str]:
    from scipy.ndimage import label
    occ = float((v > 0).mean())
    if not (0.05 <= occ <= 0.35):
        return False, f"occupancy {occ:.3f}"
    if not ((v == COL).any() and (v == SLAB).any()
            and np.isin(v, WALL_KINDS).any() and np.isin(v, PIPE_KINDS).any()):
        return False, "missing element class"
    _, n = label(v > 0)
    if n != 1:
        return False, f"{n} components"
    return True, "ok"


def _build_once(rng: np.random.Generator) -> Tuple[np.ndarray, Dict]:
    sk = build_skeleton(rng)
    strips = cut_slab_strips(sk, rng)
    add_l_borders(sk)
    add_massing(sk, rng)
    walls = add_walls(sk, rng, strips)
    add_pipes(sk, rng, walls, strips)
    add_screens(sk, rng)
    add_terrace(sk, rng)
    return sk.volume, sk.params


def generate_structure_with_params(seed: int, max_tries: int = 20) -> Tuple[np.ndarray, Dict]:
    """Deterministic per (GRAMMAR_VERSION, seed). Resamples sub-seeds on
    integrity failure — still deterministic because sub-seeds derive from seed."""
    for attempt in range(max_tries):
        rng = np.random.default_rng((seed, attempt))
        v, params = _build_once(rng)
        ok, reason = _integrity_ok(v)
        if ok:
            params["seed"] = int(seed)
            params["attempt"] = attempt
            params["occupancy"] = float((v > 0).mean())
            return v, params
    raise RuntimeError(f"seed {seed}: no valid sample in {max_tries} attempts (last: {reason})")


def generate_structure(rng_or_seed) -> np.ndarray:
    seed = rng_or_seed if isinstance(rng_or_seed, (int, np.integer)) else int(rng_or_seed.integers(0, 2**31))
    return generate_structure_with_params(int(seed))[0]
```

Note: massing runs BEFORE walls/pipes so blocks can't bury a wall/pipe count below the invariant (walls/pipes fill only empty voxels but their placement checks use geometry, not emptiness). Connectivity: slabs span the full plot and every element terminates on a slab, so a single component is the norm; the gate catches exceptions (e.g. carving isolating a shell fragment).

- [ ] **Step 4: Run the full test file, expect all pass**

```bash
uv run pytest tests/test_shodhan.py -q 2>&1 | tail -3
```

- [ ] **Step 5: Commit**

```bash
git add -u && git commit -m "feat(shodhan): deterministic assembly with integrity gates"
```

---

### Task 9: Colors tensor + dataset writer (loader-compatible layout)

**Files:**
- Modify: `deepsculpt/core/data/generation/shodhan.py`
- Test: `tests/test_shodhan.py`

The existing loaders read `<collection>/pytorch_samples/structures/structure_NNNNNN.pt` and `colors/colors_NNNNNN.pt` (see `deepsculpt/main.py:_load_sample_pairs` and `_resolve_collection_dir`), with `dataset_metadata.json` carrying `occupancy_stats` consumed by the GAN trainer.

- [ ] **Step 1: Write failing tests**

```python
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
```

- [ ] **Step 2: Run, expect failure**

- [ ] **Step 3: Implement**

```python
def write_dataset(output_dir, num_samples: int, seed_start: int = 0):
    """Write a loader-compatible collection: binary structures, element-class
    colors, per-sample params, metadata with occupancy stats + variant table."""
    import json
    from pathlib import Path

    import torch

    out = Path(output_dir)
    (out / "pytorch_samples" / "structures").mkdir(parents=True, exist_ok=True)
    (out / "pytorch_samples" / "colors").mkdir(parents=True, exist_ok=True)
    (out / "params").mkdir(parents=True, exist_ok=True)

    occupancies, variants = [], {"terrace": 0, "lattice": 0}
    level_counts: Dict[str, int] = {}
    for i in range(num_samples):
        seed = seed_start + i
        v, params = generate_structure_with_params(seed)
        structure = (v > 0).astype(np.int8)
        torch.save(torch.from_numpy(structure),
                   out / "pytorch_samples" / "structures" / f"structure_{i:06d}.pt")
        torch.save(torch.from_numpy(v.copy()),
                   out / "pytorch_samples" / "colors" / f"colors_{i:06d}.pt")
        (out / "params" / f"params_{i:06d}.json").write_text(json.dumps(params))
        occupancies.append(params["occupancy"])
        variants["terrace"] += int(params["terrace"])
        variants["lattice"] += int(params.get("screens", {}).get("lattice", False))
        lk = str(params["n_intermediate"])
        level_counts[lk] = level_counts.get(lk, 0) + 1

    occ = np.asarray(occupancies, dtype=np.float32)
    meta = {
        "grammar_version": GRAMMAR_VERSION,
        "num_samples": num_samples,
        "seed_range": [seed_start, seed_start + num_samples - 1],
        "void_dim": N,
        "occupancy_stats": {
            "mean": float(occ.mean()), "min": float(occ.min()), "max": float(occ.max()),
            "p10": float(np.percentile(occ, 10)), "p90": float(np.percentile(occ, 90)),
        },
        "variant_distribution": {**{k: v / num_samples for k, v in variants.items()},
                                 "intermediate_levels": level_counts},
    }
    (out / "dataset_metadata.json").write_text(json.dumps(meta, indent=1))
    return out
```

- [ ] **Step 4: Run tests, expect pass**

- [ ] **Step 5: Commit**

```bash
git add -u && git commit -m "feat(shodhan): loader-compatible dataset writer with params + variant metadata"
```

---

### Task 10: Dataset-level quality probe (IoU, frequency map, renders)

**Files:**
- Create: `scripts/shodhan_probe.py`
- Test: `tests/test_shodhan.py`

- [ ] **Step 1: Write failing test for the gate functions**

```python
def test_probe_gates_smoke(tmp_path):
    from scripts.shodhan_probe import probe_gates
    report = probe_gates(n=8)
    assert set(report) >= {"pairwise_iou_masked", "max_freq_outside_template", "pass"}
    assert 0.0 <= report["pairwise_iou_masked"] <= 1.0
```

Add `sys.path` note: `scripts/` is not a package; the test imports it via path. At the top of `tests/test_shodhan.py` add:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
```

- [ ] **Step 2: Run, expect failure**

- [ ] **Step 3: Implement `scripts/shodhan_probe.py`**

```python
"""Shodhan dataset probe: gates + contact sheet + interactive 3D HTML.

Usage:
    python scripts/shodhan_probe.py --n 200 --out results/shodhan_probe
Gates (spec): masked pairwise IoU < 0.45; per-voxel frequency < 0.5 outside
the slab/L-band template. Renders let the user visually approve the grammar
on REAL 64^3 samples before full generation.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from deepsculpt.core.data.generation import shodhan as sh


def probe_gates(n: int = 200, seed_start: int = 0) -> dict:
    vols, temps = [], []
    for i in range(n):
        v, _ = sh.generate_structure_with_params(seed_start + i)
        vols.append(v > 0)
        temps.append((v == sh.SLAB) | (v == sh.EDGE))
    vols = np.stack(vols)
    template = np.stack(temps).mean(0) > 0.95        # near-constant slab/L voxels
    freq = vols.mean(0)
    max_freq_out = float(freq[~template].max())

    ious = []
    idx = np.random.default_rng(0).permutation(n)[:min(n, 60)]
    for a, b in combinations(idx, 2):
        va, vb = vols[a] & ~template, vols[b] & ~template
        union = (va | vb).sum()
        ious.append((va & vb).sum() / union if union else 0.0)
    iou = float(np.mean(ious))

    report = {
        "n": n,
        "pairwise_iou_masked": iou,
        "max_freq_outside_template": max_freq_out,
        "occupancy_mean": float(vols.mean()),
        "pass": iou < 0.45 and max_freq_out < 0.5,
    }
    return report


def render_reports(n_render: int, out_dir: Path, seed_start: int = 0) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go

    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(seed_start, seed_start + n_render))
    # contact sheet: projections + mid slice
    fig, axes = plt.subplots(len(seeds), 4, figsize=(11, 2.6 * len(seeds)), squeeze=False)
    vols = []
    for r, s in enumerate(seeds):
        v, p = sh.generate_structure_with_params(s)
        vols.append(v)
        b = v > 0
        for c in range(3):
            axes[r][c].imshow(b.max(axis=c), cmap="gray_r", vmin=0, vmax=1)
            axes[r][c].set_xticks([]); axes[r][c].set_yticks([])
        axes[r][3].imshow(b[:, :, 32], cmap="gray_r", vmin=0, vmax=1)
        axes[r][3].set_xticks([]); axes[r][3].set_yticks([])
        axes[r][0].set_ylabel(f"s{s} occ {b.mean():.3f}", fontsize=8)
    fig.suptitle("shodhan v2 probe — projections + mid slice")
    fig.tight_layout()
    fig.savefig(out_dir / "contact_sheet.png", dpi=110)
    plt.close(fig)

    # interactive 3D html (first 3 seeds), colored by element class
    palette = {sh.COL: "#5a5a5a", sh.SLAB: "#c8c4bc", sh.SCREEN: "#e4dccb",
               sh.PIPE_R: "#c0392b", sh.PIPE_B: "#2471a3", sh.PIPE_Y: "#f1c40f",
               sh.VOL: "#a8a29a", sh.EDGE: "#f7f6f2", sh.WALL_R: "#b5493a",
               sh.WALL_B: "#3a6bb5", sh.WALL_Y: "#e0b839", sh.WALL_G: "#6b8e4e"}
    figp = go.Figure()
    for i, v in enumerate(vols[:3]):
        x, y, z = np.where(v > 0)
        cols = [palette.get(int(k), "#999") for k in v[x, y, z]]
        figp.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="markers",
                                    marker=dict(size=2, symbol="square", color=cols),
                                    name=f"seed {seeds[i]}", visible=(i == 0)))
    figp.update_layout(
        updatemenus=[dict(buttons=[dict(label=f"seed {seeds[i]}", method="update",
                                        args=[{"visible": [j == i for j in range(min(3, len(vols)))]}])
                                   for i in range(min(3, len(vols)))], x=0.02, y=0.98)],
        scene=dict(aspectmode="cube", xaxis=dict(visible=False),
                   yaxis=dict(visible=False), zaxis=dict(visible=False)),
        title="shodhan v2 probe — drag to rotate", height=800)
    figp.write_html(out_dir / "probe_3d.html", include_plotlyjs="cdn")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--n-render", type=int, default=8)
    ap.add_argument("--out", default="results/shodhan_probe")
    args = ap.parse_args()
    out = Path(args.out)
    report = probe_gates(args.n)
    render_reports(args.n_render, out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "gates_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    print(f"renders -> {out}/contact_sheet.png, {out}/probe_3d.html")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test + a tiny live probe**

```bash
uv run pytest tests/test_shodhan.py::test_probe_gates_smoke -q 2>&1 | tail -2
python scripts/shodhan_probe.py --n 16 --n-render 4 --out /tmp/shodhan_probe_smoke
```
Expected: test passes; probe prints a JSON report (pass may be true or false at n=16 — thresholds are calibrated at n=200).

- [ ] **Step 5: Commit**

```bash
git add scripts/shodhan_probe.py tests/test_shodhan.py
git commit -m "feat(shodhan): quality probe — masked IoU + frequency gates + renders"
```

---

### Task 11: CLI wiring — `generate-data --structure-preset shodhan`

**Files:**
- Modify: `deepsculpt/main.py` (the `generate_data` handler and the `generate-data` argparse section; find them via `grep -n "generate-data\|def generate_data" deepsculpt/main.py`)
- Test: `tests/test_shodhan.py`

- [ ] **Step 1: Write failing test**

```python
def test_cli_generate_shodhan(tmp_path):
    import subprocess, sys, json
    r = subprocess.run(
        [sys.executable, "-m", "deepsculpt.main", "generate-data",
         "--num-samples", "3", "--void-dim", "64",
         "--structure-preset", "shodhan", "--output-dir", str(tmp_path)],
        capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    metas = list(tmp_path.rglob("dataset_metadata.json"))
    assert metas, "no collection written"
    meta = json.loads(metas[0].read_text())
    assert meta.get("grammar_version") == "2.0.0"
```

- [ ] **Step 2: Run, expect failure** (unknown preset choice or old generator path)

- [ ] **Step 3: Implement.** In `deepsculpt/main.py`:

(a) In the `generate-data` parser section, extend the preset choices — find the existing `--structure-preset` argument (grep `structure-preset`; if it only exists as `getattr(args, "structure_preset", ...)`, add the argument to the generate-data parser):

```python
    gen_data_parser.add_argument('--structure-preset', default='shodhan',
                                choices=['shodhan', 'architectural', 'generic'],
                                help='shodhan = Corbusier/Umemoto grammar v2 (default); architectural/generic = legacy')
```

(b) In the `generate_data` handler, route the shodhan preset before the legacy collector path (adapt names to the actual handler; keep the legacy branch untouched):

```python
        if getattr(args, 'structure_preset', 'shodhan') == 'shodhan':
            from datetime import date
            from deepsculpt.core.data.generation.shodhan import write_dataset
            out = Path(args.output_dir) / date.today().isoformat()
            write_dataset(out, num_samples=args.num_samples,
                          seed_start=getattr(args, 'seed_start', 0))
            print(f"Dataset generated successfully! Collection directory: {out}")
            return 0
```

Also add `--seed-start` (type int, default 0) to the generate-data parser so the 20k train and 1k held-out sets use disjoint seed ranges.

- [ ] **Step 4: Run test, expect pass**

```bash
uv run pytest tests/test_shodhan.py::test_cli_generate_shodhan -q 2>&1 | tail -2
```

- [ ] **Step 5: Full test file + nano CPU sanity of the training loader against a shodhan collection**

```bash
uv run pytest tests/test_shodhan.py -q 2>&1 | tail -2
python -m deepsculpt.main generate-data --num-samples 8 --void-dim 64 --structure-preset shodhan --output-dir /tmp/shodhan_smoke
python -m deepsculpt.main train-gan --model-type skip --discriminator-type spectral_norm --gan-loss-type softplus --void-dim 64 --epochs 1 --batch-size 2 --data-folder /tmp/shodhan_smoke --output-dir /tmp/shodhan_train_smoke --num-workers 0
```
Expected: 1 nano epoch completes on the old machine (this is a loader-compatibility check, NOT training).

- [ ] **Step 6: Commit**

```bash
git add -u && git commit -m "feat(cli): shodhan preset is the generate-data default"
```

---

### Task 12: Docs

**Files:**
- Modify: `CLAUDE.md` ("Architectural data generator is the active mode" bullet in Current status; the file-structure tree line for `core/data/generation`)
- Modify: `README.md` only if it documents generate-data flags (grep `generate-data README.md`; if absent, no change)

- [ ] **Step 1: CLAUDE.md surgical edits**

Replace the bullet
`- **Architectural data generator is the active mode** — columns + slabs + 3 orthogonal pipes (red/blue/yellow). See recent \`git log\` for the procedural-shape tuning history.`
with:

```markdown
- **Shodhan grammar v2 is the active data generator** (`--structure-preset shodhan`, default): Corbusier Dom-ino skeleton × Umemoto carved massing at 64³ — full-square slabs (the one template element), structured 3×3/3×4/4×4 column arrays, strip-cut intermediate slabs, column-to-column polychrome walls (2/level, some double-height), flush vertical-fin brise-soleil, wall-anchored pipes, carved blocks (1-voxel shells, slab-to-slab openings), roof terraces with pavilion block. Deterministic per seed; per-sample `params_*.json`; spec at `docs/superpowers/specs/2026-07-05-shodhan-dataset-design.md`. Probe before GPU spend: `python scripts/shodhan_probe.py --n 200` (gates + contact sheet + 3D HTML). Legacy presets kept: `architectural`, `generic`.
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md README.md 2>/dev/null; git add CLAUDE.md
git commit -m "docs: shodhan grammar v2 is the active data generator"
```

---

### Task 13: OPS — probe, visual gate, full generation, baseline training

No new code. Exact commands; the **visual gate requires the user**.

- [ ] **Step 1: Full probe locally (CPU, ~minutes)**

```bash
python scripts/shodhan_probe.py --n 200 --n-render 8 --out results/shodhan_probe
```
Expected: `"pass": true` in the report. If gates fail, tune the ⚙ constants in `shodhan.py` (mass dial range, screen probability) and re-run — do NOT loosen the thresholds without the user.

- [ ] **Step 2: VISUAL GATE — user approval (BLOCKING)**

Show the user `results/shodhan_probe/contact_sheet.png` and `results/shodhan_probe/probe_3d.html`. Proceed only on explicit approval; expect rule tweaks (the ⚙ constants exist for this).

- [ ] **Step 3: Merge branches** (user action or with approval): PR for `fix/gan-bf16-resume` (#7) then `feat/shodhan-dataset` (spec + this implementation). Image rebuild on master push publishes the new generator to GAR/GHCR.

- [ ] **Step 4: Generate the corpus to GCS.** 20k train (seeds 0–19999) + 1k held-out (seeds 100000–100999). Two options — local (hours OK on any machine, it's CPU numpy) or a Cloud Run execution reusing the job with `TRAIN_CMD` unused and datagen driven by the entrypoint's cache-miss path. Recommended: run the two commands locally or on any CPU box with gcloud auth:

```bash
python -m deepsculpt.main generate-data --num-samples 20000 --void-dim 64 \
  --structure-preset shodhan --seed-start 0 --output-dir /tmp/shodhan_v2_train
python -m deepsculpt.main generate-data --num-samples 1000 --void-dim 64 \
  --structure-preset shodhan --seed-start 100000 --output-dir /tmp/shodhan_v2_holdout
gsutil -m rsync -r /tmp/shodhan_v2_train  gs://garassino-ml-artifacts/deepsculpt/data/void64-shodhan-v2/train
gsutil -m rsync -r /tmp/shodhan_v2_holdout gs://garassino-ml-artifacts/deepsculpt/data/void64-shodhan-v2/holdout
```

- [ ] **Step 5: Point the Cloud Run data cache at the new set.** The entrypoint pulls `data/void${VOID_DIM}` — for v2, pass `GCS_DATA` override or simply sync the new train set into the keyed path after archiving the old one:

```bash
gsutil -m rsync -r gs://garassino-ml-artifacts/deepsculpt/data/void64 gs://garassino-ml-artifacts/deepsculpt/data/void64-archifacts-v1
gsutil -m rm -r gs://garassino-ml-artifacts/deepsculpt/data/void64
gsutil -m rsync -r gs://garassino-ml-artifacts/deepsculpt/data/void64-shodhan-v2/train gs://garassino-ml-artifacts/deepsculpt/data/void64
```

- [ ] **Step 6: Baseline GAN (proven cr-005 recipe + ada-lite)**

```bash
gcloud run jobs execute deepsculpt-train --project garassino-ml --region europe-west1 \
  --update-env-vars "^@^RUN_ID=gan-shodhan-001@TRAIN_CMD=train-gan@PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True@TRAIN_ARGS=--model-type skip --discriminator-type spectral_norm --void-dim 64 --epochs 100 --batch-size 16 --mixed-precision --num-workers 4 --gan-loss-type softplus --ttur-ratio 1.0 --augment ada-lite --augment-target 0.7 --resume" --async
```
Monitor with the session's monitor pattern (GCS `snapshots/epoch_*.json` + Cloud Logging errors). Kill criteria: non-finite metrics, fakeOcc pinned at 0/1, G > 1e4 sustained. Chain 1h slices with the same command (`--resume` continues). At epoch ~30: render `python scripts/render_run_snapshots.py gan-shodhan-001` + raw-vs-EMA check + pairwise IoU vs the dataset's own masked IoU (±0.15 target).

- [ ] **Step 7: Baseline diffusion (proven b4 recipe)**

```bash
gcloud run jobs execute deepsculpt-train --project garassino-ml --region europe-west1 \
  --update-env-vars "^@^RUN_ID=diff-shodhan-001@TRAIN_CMD=train-diffusion@PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True@TRAIN_ARGS=--void-dim 64 --epochs 100 --batch-size 4 --model-channels 64 --grad-checkpoint --mixed-precision --scheduler --resume" --async
```
Monitor `logs/epoch_metrics.jsonl` in GCS: loss must decrease epoch-over-epoch; at epoch ~30 pull a checkpoint and sample (DDIM) for a visual check.

- [ ] **Step 8: Report to user** — renders of both models on the new data; decision point for the latent-track plan (separate plan per spec Stage 2).

---

## Self-review notes

- Spec coverage: E1–E3 (Task 1), C1/S3 (Task 2), W1–W6 (Task 3), P1–P4 (Task 4), F1–F2 (Task 5), M1–M2 (Task 6), T1–T2 (Task 7), G1–G4 + determinism (Tasks 8–9), gates + visual gate (Tasks 10, 13), CLI + default preset (Task 11), docs (Task 12), baseline training (Task 13). Stage-2 latent: separate plan by design.
- The mockup file remains the visual reference; where mockup and plan differ (stairs removed, params capture, integrity resampling, 64³ constants) the plan wins.
- Type consistency: `Skeleton`, `Wall`, and function signatures are used consistently across tasks; `generate_structure_with_params` is the single public entry point used by writer, probe, and CLI.
