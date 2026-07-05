# Shodhan dataset v2 — design spec

**Date:** 2026-07-05
**Status:** approved-pending-review
**Reference mockups:** `.superpowers/brainstorm/mockups_v4.py` (grammar v15, validated visually over 15 iterations)
**Supersedes:** the "architectural" preset (fixed grid + 3 planes + 2 pipes), whose fixed template
was measured to make up 69% of a typical sample and drove GAN mode collapse.

## Goal

Replace the DeepSculpt training dataset with a procedurally generated corpus in a
Corbusier/Umemoto architectural grammar ("Villa Shodhan rules"): a fixed structural
template (slabs on a plot) carrying high parametric variety (grids, walls, screens,
pipes, carved massing, terraces), engineered so 3D GAN and diffusion models have
real structure to learn and measurable diversity to reproduce.

Resolution: **64³, generated directly** (no multi-resolution pipeline for now).
Training: **monochrome first** (structure tensor); the colors tensor ships as
per-element semantic labels for later conditional/weighted training.

## The grammar (all values at 64³; mockup values ×2; marked tunable ⚙ where the
visual gate may adjust them)

Element classes (encoded in the colors tensor): column, slab, L-border/edge,
wall (4 polychrome colors), screen, pipe (red/blue/yellow), massing volume.

### Envelope — the sanctioned template
- **E1** Every building occupies the identical envelope: plot square with margin 4
  (i.e. [4..59]²) × full void height. All samples share it.
- **E2** Slabs (thickness 2 ⚙) always span the full plot square. Ground slab and
  last slab always intact.
- **E3** Storey height sampled from {14, 16, 20, 26} with weights
  {0.35, 0.35, 0.2, 0.1} → 1–3 intermediate slabs
  (14 → 3 intermediates, 16/20 → 2, 26 → 1).

### Structure
- **S1** Column array: evenly distributed 3×3 / 3×4 / 4×4 (uniform choice per axis
  from {3,4}); inset from plot edge 4–8 per side; insets chosen so column spacing is
  exactly uniform, symmetric insets preferred. Columns 2×2 thick ⚙.
- **S2** Columns run ground → last slab. Never on the terrace level.
- **S3** L-border: a 1-voxel white band on every slab rim, all four sides,
  directly above AND below the slab — a continuous ring, never broken.

### Slab cuts
- **C1** Intermediate slabs only: one strip removed per slab, spanning the full
  plot between two adjacent column rows (strictly between the rows), preserving
  the rim ring (outermost slab voxels). Registry of strips is input to wall/pipe
  placement.

### Walls
- **W1** Exactly two walls per level, always between consecutive slabs.
- **W2** Walls lie on a column grid line with endpoints at columns, spanning ≥ 2 bays.
- **W3** Walls keep ≥ 6 voxels clearance from the plot edge.
- **W4** Walls never stand on a removed strip; single-storey walls also avoid
  strips in their ceiling slab. Placement is resample-until-valid (24 tries).
- **W5** With p=0.3 per eligible level, the second wall is double-height: spans two
  levels, stands on solid slab, may rise through the strip void or the intact slab.
- **W6** Wall thickness 2 ⚙; color per wall sampled from the polychrome palette
  {red, blue, yellow, green}.

### Facade screens (brise-soleil)
- **F1** Present on 75% of buildings, on 1–2 facades; flush with the slab rim plane
  (possible because cantilever/inset is constant per side); 2 voxels deep ⚙;
  vertical span ground slab → last slab, terminating at slabs.
- **F2** Vertical fins only (bar width 2 ⚙, rhythm 6–8 ⚙) — EXCEPT buildings with a
  single intermediate slab, which may use the full lattice (vertical + horizontal
  bars) with p=0.7.

### Pipes
- **P1** Always 2–3 pipes: a vertical riser (2×2 ⚙) hugging a wall face, from the
  ground slab to the slab its wall serves, plus a horizontal run under that slab.
- **P2** Risers never pass through strip voids of any slab they cross; horizontal
  runs are clipped at strip boundaries. Resample-until-valid (30 tries).
- **P3** Nothing out: pipes strictly inside the facade plane (rim reserved for
  slab/L-band/screen).
- **P4** Colors: saturated red / blue / yellow, shuffled per sample.

### Massing and carving (volume and emptiness)
- **M1** Mass dial m ∈ [0.25, 0.85] per sample → 1–4 volume blocks (footprint
  16–32 ⚙ scaled by m), corners snapped near column lines, sitting on ground/slabs,
  1–2 levels tall.
- **M2** Every block is carved — this is where the emptiness comes from:
  hollow core (interior room, shell 2–4 thick, with a door slot), through-tunnel
  piercing the block (p=0.8), 1–2 corner notches. **All openings are full-height,
  slab to slab — abstracted doors** (vertical slots, never partial-height holes).
  Re-carve until the block is ≤ 70% solid within its own bounding box.
- **M3** Stairs: 1–2 stepped diagonals (width 4–6 ⚙) connecting consecutive slabs.

### Terrace
- **T1** Only buildings with 3 intermediate slabs may omit the roof slab (p=0.45).
- **T2** Terrace level: NO columns. Carries exactly ONE carved massing block
  (footprint 20–32 ⚙, M2 carving rules, rising to the roof plane so the envelope
  height always reads) plus exactly ONE accent wall, and a pipe rising to the
  roof plane (p=0.5).

### Global integrity invariants
- **G1** Nothing floats: every element terminates on a slab or the ground; each
  sample must be a single connected component (checked; reject-and-resample).
- **G2** Walls, screens, and edges never occupy the same plane.
- **G3** Columns + walls + pipes present in every sample.
- **G4** The colors tensor encodes element class per voxel (semantic labels).

## Architecture

New module: `deepsculpt/core/data/generation/shodhan.py`
- Pure function `generate_shodhan(seed: int, void_dim: int = 64) -> tuple[np.ndarray, np.ndarray, dict]`
  returning (structure int8 binary, colors, params dict with every sampled value).
  Deterministic per (code version, seed). The mockup file is the reference
  implementation to port; production code reuses existing primitives
  (`pytorch_shapes.py`) only where they fit — the grammar logic is new.
- Wire into the CLI: `generate-data --structure-preset shodhan` becomes the
  default; the old "architectural" preset remains available (legacy).
- Per-sample outputs (same layout the loaders already read):
  `structure_NNNNNN.pt`, `colors_NNNNNN.pt`, plus new `params_NNNNNN.json`.
- Collection metadata gains: grammar version, seed manifest, occupancy stats
  (already consumed by the GAN trainer's occupancy penalty), variant distribution
  (terrace %, lattice %, array sizes, level counts).

## Quality gates (extend ds-dataval; run before any GPU spend)

Per-sample (reject-and-resample, logged):
- occupancy ∈ [0.05, 0.35] ⚙
- single connected component
- element presence: ≥1 column array, ≥2 walls, ≥2 pipes

Dataset-level (200-sample probe, auto-report):
- pairwise IoU with slab template voxels masked out: target < 0.45
- per-voxel frequency < 0.5 outside slab planes and L-bands
- occupancy histogram; variant distribution table
- auto-generated contact sheet (projections + mid-slices) and interactive 3D HTML
  (plotly, as used in this session) from the first 100 samples

**Visual gate:** the user reviews the contact sheet + 3D HTML and approves the
dataset before generation of the full corpus / any training.

## Scale, storage, determinism

- 20,000 train + 1,000 held-out samples, generated from a committed seed manifest.
- Storage: int8 tensors as today; GCS at
  `gs://garassino-ml-artifacts/deepsculpt/data/void64-shodhan-v2/`.
- Generation is CPU-only; runs locally for the probe and as a CPU Cloud Run
  execution for the full corpus (the entrypoint's datagen path, ~minutes).

## Training (first round, after the gates pass)

- Monochrome structure only. GAN: the proven gan-cr-005 recipe
  (skip generator + spectral_norm discriminator + softplus + bf16 +
  `--augment ada-lite` to hold diversity). Diffusion: batch 4 + grad-checkpoint +
  model-channels 64 (the proven-fitting config).
- Success criteria: GAN sample pairwise IoU tracks the dataset's own value
  (±0.15) through epoch 60; generated samples show slabs/columns/walls/pipes
  recognizable in renders; diffusion loss decreasing with samples developing
  slab planes by epoch ~30.

## Out of scope (separate follow-up specs)

- Stage-2 latent track: light-KL VAE 64³→16³×8 (GroupNorm, BCE pos_weight +
  soft-Dice, class-weighted via colors), then latent diffusion + latent GAN,
  reconstruction IoU ≥ 0.95 gate. Enables higher resolution later.
- Conditional training on params.json; colored/semantic-channel training.
- TSDF / soft-occupancy representation.

## Docs to update (same PR as implementation)

- `CLAUDE.md`: "Architectural data generator" bullet → Shodhan grammar summary +
  preset flag; dataset GCS path; visual-gate workflow.
- `docs/` (data docs if present): grammar rule list.
- `README.md`: only if generate-data CLI usage changes (flag rename).
