# DeepSculpt — Color Latent Diffusion with Continuous Palettes (RGBA)

## Context

The latent diffusion stack shipped and verified today (VAE IoU 0.9998 in one epoch;
latdiff at 7.5 min/epoch full-data on an L4). Juan wants **color diffusion** — and instead
of the GAN's 13-class categorical field, the natural diffusion representation is a
continuous **RGBA voxel field** (diffusion is built for continuous channels; this is the
3D analog of image diffusion, and the project's ancestral v1 data format). Because RGB is
continuous, the palette no longer needs to be 12 flat hex colors: Juan explicitly wants
**pastels and gradients** — procedurally generated per-sample color schemes (seeded,
deterministic) that make the learned color distribution rich and pretty. Matplotlib and
the three.js viewer both take per-face/per-instance RGB directly.

Key architectural facts being reused (all shipped today, canary-verified):
- `VAE3D(in_channels, latent_channels, base_channels)` — fully-conv KL-VAE
  (`core/models/autoencoder/vae3d.py`), `AutoencoderTrainer` (BCE+KL, IoU gate),
  `LatentCodec` (deterministic per-channel shift/scale from an unshuffled 256-sample
  prefix, asserted on resume), `train-diffusion --latent-autoencoder` (latdiff_* runs,
  `LatentFastSamplingPipeline` decodes everywhere via `_finalize`, clip_sample off),
  loader latent branch, VAE staged under the latdiff RUN_ID's `vae/` GCS dir.
- Batch dicts already carry `index` (`main.py:116-141`) — free deterministic per-sample
  palette seeds. `colors` = int class indices 0-12 (legacy [B,1,64³] squeeze needed).
- Stage-A diffusion recipe (v-pred + min-SNR 5 + cosine) verified — reuse for color.

## Design

### 1. Procedural palette module — `core/data/transforms/palette.py` (new)

`build_rgba(structure, colors, indices, mode, base_seed=0) -> [B, 4, 64³] float [0,1]`
Channel order **[alpha, R, G, B]** — channel 0 stays "presence", so every existing
threshold-channel-0 convention survives.

Determinism rules (critique-hardened):
- Seed mix is an explicit integer op — `seed = (base_seed * 1_000_003 + int(index)) & 0x7FFFFFFFFFFFFFFF`
  into a dedicated CPU `torch.Generator`; never `hash()`, never the global RNG.
- **Fixed draw layout**: every sample draws 12 classes × k params in class order 1..12
  regardless of which classes appear — content never shifts the RNG stream, and all
  modes share one layout (mode changes never re-scramble draws).
- Per-sample param table `[12, 2 endpoints, 3]` drawn on CPU; endpoint colors generated
  in HSV (hue ±0.06, sat clamped [0.15, 0.5], value [0.70, 0.95], anchored to
  CLASS_PALETTE base hues), converted to RGB once per sample; per-voxel z-lerp happens
  in RGB (endpoints share a hue, so RGB lerp ≈ perceptual; no wraparound handling).
- `palette_version` int stamped in configs so future math changes are explainable.
- Streaming-fallback datasets (no `colors`/`index` keys) raise in RGBA mode — no
  silent degradation.
- **Ownership: palette params (mode/seed/version) live ONLY in the VAE's config.json**
  (written by train-autoencoder) and are inherited by train-diffusion via `load_vae`'s
  config — there is NO palette CLI on train-diffusion (a VAE is trained on exactly one
  palette; letting diffusion pick another is never valid). The existing shift/scale
  resume assertion doubles as the drift canary.

Modes (user decision: subtle AND bold both implemented; first color VAE trains on subtle;
critique preferred cutting gradients from v1 — overridden by explicit user ask, mitigated
by local PNG render of all modes before any cloud spend):
- `flat` — exact CLASS_PALETTE lookup (baseline/debug; palette constants move to this
  module, `render_walk.py` imports them back — single source of truth).
- `subtle` — Shodhan-cohesive: each class's base color pastelized per sample in HSV
  (hue ±0.04, saturation ×U(0.35, 0.7), value →U(0.8, 0.95)) plus per-class vertical
  gradient (two per-sample endpoint variants lerped by normalized z, endpoints within
  ±0.12 value — reads as light, not rainbow). Dedicated `torch.Generator` seeded
  `base_seed*1e6 + index` — never the global RNG (determinism across slices/workers).
- `bold` — subtle + a per-sample global hue rotation U(0, 1) applied to the accent
  classes (pipes/walls) while concrete neutrals (COL/SLAB/VOL/EDGE) keep low saturation:
  blue-dominant, terracotta, green sculptures — wide color distribution, architectural
  identity preserved by the neutral structure classes.

Empty voxels get RGB = 0.5 (neutral) — not black — so unmasked-region targets sit at the
sigmoid midpoint and contribute ~zero gradient pressure.

### 2. RGBA VAE — `train-autoencoder --rgba --palette gradient`

- `VAE3D(in_channels=4, latent_channels=8)` (color needs more latent capacity than
  binary; 16³×8 still tiny for the UNet).
- CLI: a single `--palette {flat,subtle,bold}` flag implies RGBA (no separate `--rgba` —
  kills inconsistent-state combos) + `--palette-seed`.
- `AutoencoderTrainer` gains `palette_cfg: Optional[PaletteConfig]`; when set, the input
  builder produces ARGB via `build_rgba` (raises if `colors`/`index` missing).
- Loss: `BCE(alpha_logits, alpha) + rgb_weight × masked_MSE(sigmoid(rgb_logits), rgb)`
  masked by **ground-truth** alpha > 0.5 (mean over masked elements, `clamp_min(1)`
  denominator). `--rgb-weight` default **1.0** (BCE ≈ 0.69 and masked MSE ≈ 0.04-0.08 at
  init — same order; escalate to 3-5 only if val MAE plateaus above gate). Empty-voxel
  RGB targets/inputs filled with **0.5** (neutral — 0-fill causes dark rims at occupancy
  boundaries).
- `validate()`: alpha IoU (gate ≥ 0.95) + masked RGB MAE (gate < 0.05) → epoch_metrics.
- Recon snapshots: ARGB fp16 stacks (renderable once §4 lands).
- config.json gains `rgba: true`, `palette_mode`, `palette_base_seed`.

### 3. Color latent diffusion — reuse everything

- `DiffusionTrainer` gains `latent_input_fn: Optional[Callable[[dict], Tensor]]` — a
  closure over the PaletteConfig built in `main.train_diffusion` (codec stays pure, the
  loader/decode path untouched); latent path: `x_0 = codec.encode(latent_input_fn(batch)
  if latent_input_fn else structure.unsqueeze(1))`.
- `main.train_diffusion` derives RGBA from `vae.in_channels == 4`, reads palette params
  from the VAE config (hard error if in_channels==4 but palette fields absent — catches
  stale staged configs), builds the 256-sample stats prefix with `build_rgba`, records
  the palette dict in the latdiff config.json latent block.
- **Fix two pre-existing/silent bugs in the same commit**: (1) `_export_latent_outputs`
  (`main.py:1195-1199`) argmaxes any C>1 volume into fake class ids — decoded RGBA
  latent-walks would render garbage through the class palette; add an `rgba` branch
  (geometry from ch0, save full RGBA .pt). (2) `diffusion_final.pt` (`main.py:680-694`)
  writes `void_dim=args.void_dim` (64) and omits the `latent` block — the final export
  of any latent run would rebuild as a pixel-mode UNet; write latent-aware fields.
- UNet: existing latent arch (`void_dim=16`, `num_channels=8` from config, mult [1,2,4],
  attn [8,4]) — zero UNet changes.
- Training recipe: the verified Stage-A flags (`--prediction-type v_prediction
  --min-snr-gamma 5 --noise-schedule cosine`), batch 64, L4.

### 4. RGB rendering

- `render_walk.py`: detect RGBA (`ndim==5 and shape[1]==4`) BEFORE the
  `reshape(N, *shape[-3:])` (which would otherwise error); voxel branch — occupancy from
  ch0 > threshold, per-face colors from ch1:4 at exposed voxels × directional shade;
  mesh style falls back to alpha + mono cmap (documented).
- `walk_viewer.py`: pack 6×int16 (x,y,z,r,g,b) + `format` field in the HTML payload;
  parametrize the JS stride/`maxCount` (currently hardcoded `/8`); `setColorAt(rgb/255)`.
- `render_run_snapshots.py`: `[:, 0]` guard when 5D (one line, before reshape).
- `_snapshot_sample_stats` (`diffusion_trainer.py:540-548`): occupancy from channel 0
  when 5D and C > 1 — must ship BEFORE the first color slice (occupancy telemetry
  otherwise reads pastel RGB means, ~0.6-0.8, silently wrong).
- `volume_export._to_volume` already collapses (4,64³) to the alpha channel via `v[0]` —
  geometry-only OBJ/STL/gif exporters keep working unchanged (verified by critique);
  [A,R,G,B] channel order is what makes this true. `AutoencoderTrainer` recon snapshots
  save `[2N,4,64³]` — renderable once the render_walk RGBA branch lands.

### 5. Cloud sequencing (all under the auto-shutdown)

1. Nano CPU tests → commit → `cloudbuild-cu126.yaml` build → update L4 job image.
2. `vae-color-001` canary (L4, **batch 8** — RGBA first-conv width unchanged, memory ≈
   binary VAE; `--palette subtle --latent-channels 8`, checkpoint-freq 1) → gate:
   val alpha-IoU ≥ 0.95 AND `val_rgb_mae` < 0.05; budget 2-4 slices, inspect recon
   snapshots at class contacts before gating.
3. Stage VAE checkpoint **plus its config.json (palette fields!)** under
   `gs://…/checkpoints/latdiff-color-001/vae/`; launch `latdiff-color-001` canary →
   `deepsculpt-latdiff-color-chain` scheduler (every 3h, offset from existing chains;
   covered by `deepsculpt-auto-shutdown` — note deadline currently 2026-07-14 06:00 UTC;
   user extends it explicitly if color training should continue past it).
4. First rendered verdict: decoded RGBA walk via updated `render_walk --style voxel` +
   viewer HTML.

## Risks / mitigations (from adversarial critique)

| Risk | Mitigation |
|---|---|
| Palette params drift across resumed slices | Params live only in the VAE config, inherited; fixed 12-class draw layout; shift/scale resume assertion as backstop |
| Stale binary-VAE config staged for a color run → silently trains structure-only | Hard error when `vae.in_channels==4` but palette fields absent; RGBA banner logged at start |
| RGBA latent-walk renders garbage (argmax bug) | `_export_latent_outputs` fix ships in the same commit as the trainer path |
| Muddy colors at class contacts (4× spatial compression mixes classes in a block) | Inspect recon snapshots at pipe/slab contacts before gating; escalation: rgb_weight ↑ → latent 12ch. Pastels hide muddiness well |
| RGBA VAE gate slower than the binary one-epoch gate | Budget 2-4 L4 slices; chain resumes; gate read from epoch_metrics.jsonl |
| Occupancy telemetry silently wrong on RGBA | `_snapshot_sample_stats` ch0 fix ships before the first color slice |
| PIPE_R/WALL_R share a base hex | Per-(sample, class-slot) jitter makes them diverge per sample — intended; covered by a nano test |

## v1 scope cuts

- No RGBA pixel-space diffusion (latent only). No mesh-style RGBA rendering (alpha+mono fallback).
- No colored OBJ/STL export (geometry-only via alpha comes free from `_to_volume`).
- No `sample-diffusion --visualize` / PyTorchVisualizer RGBA support.
- No per-class semantic recovery from RGB (visual fidelity is the goal).
- The GAN keeps its 13-class pipeline untouched.
- Existing `train-diffusion --color` 6-channel placeholder: replaced by a clear error
  pointing at `--latent-autoencoder` with an RGBA VAE.

## Verification

- Nano (CPU) — `tests/test_palette.py` + inline checks: build_rgba determinism (same
  index twice ⇒ bit-identical; stable across Subset and legacy [B,1,64³] colors); fixed
  draw layout (sample with 3 classes present == same colors as one with 12); channel 0
  == structure exactly; `flat` == CLASS_PALETTE; HSV bounds respected; PIPE_R vs WALL_R
  diverge per sample; RGBA VAE fwd/bwd at 8³ (fully conv); masked-loss gradient exactly
  zero on empty voxels; codec round-trip in_channels 4; latent train_step via
  latent_input_fn; decoded sample (1,4,64³) in [0,1]; **render one PNG per palette mode
  from grammar samples and eyeball locally BEFORE any cloud spend**.
- Cloud canaries: checkpoint-in-GCS standard; VAE gate from epoch_metrics.jsonl;
  latdiff-color epoch-1 decoded snapshot renders in true color.

## Docs to update (same commits)

- CLAUDE.md: palette module + modes, `--rgba --palette` flags, RGBA channel order
  [A,R,G,B], color-latent recipe, renderer RGBA support.
- Spec: `docs/superpowers/specs/2026-07-13-color-latent-diffusion-design.md`.
