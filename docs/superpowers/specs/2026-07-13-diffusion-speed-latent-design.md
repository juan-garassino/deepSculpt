# DeepSculpt — Faster Diffusion + Latent Diffusion (Stages A/0/B/C)

## Context

The color GAN produces genuinely architectural sculptures (epoch 33+); pixel diffusion,
after two root-cause fixes (torch-2.13 bf16 autocast `641b332`, [-1,1] zero-centering
`eda0230`), trains but converges slowly (21 min/epoch on the RTX at 12k samples; density
still ~5× too high after 5 epochs). Juan's priorities in order: **(A) faster convergence
for the current pixel diffusion, (B) latent diffusion for speed+quality at 64³, (C) keep
a fully-conv path to 128³.** GAN slab-count preference is parked (reference PNGs in
`~/Downloads/deepsculpt-progress/slab-count-samples/`; levers documented below).

Standing constraints: GCP-only; every scheduler under the auto-shutdown deadline;
1h task-timeouts (an epoch MUST fit a ~50-min slice); GCP-native builds
(`infra/gcp/cloudbuild-cu128.yaml` for RTX, GHA for the cu126/GHCR mirror);
canary = checkpoint-in-GCS, never logs; nano CPU tests only locally.

---

## Stage A — pixel-diffusion convergence package (`diff-shodhan-007`, RTX)

Bundle three training-dynamics upgrades into ONE restart. Existing code already provides:
cosine schedule (`noise_scheduler.py:85-93`), `get_velocity` (`:145-171`), v-pred training
target (`diffusion_trainer.py:308-316`), v-pred reconstruction in base `NoiseScheduler.step`
(`:201-205`), prediction_type plumbing trainer→pipelines (`diffusion_trainer.py:52,79,104,552`).

Edits:
1. **DDIMScheduler.step v_prediction branch** (`noise_scheduler.py` ~:325):
   `pred_x0 = √ᾱ[t]·sample − √(1−ᾱ)[t]·model_output` (mirror :201-205). DDIM is the
   default sampler for snapshots/walks — mandatory.
2. **DPMSolverScheduler.step v_prediction branch** (~:417, prediction_type is an instance
   attr at :375).
3. **Min-SNR weighting in `DiffusionTrainer.compute_loss`** (`diffusion_trainer.py:137-187`):
   per-sample MSE (`reduction="none"`, mean over non-batch dims) weighted by
   `w = min(SNR,γ)/SNR` (epsilon) or `min(SNR,γ)/(SNR+1)` (v-pred), `SNR = ᾱ/(1−ᾱ)` from
   `noise_scheduler.alphas_cumprod[timesteps]`. New trainer ctor param
   `min_snr_gamma: Optional[float] = None`.
4. **CLI/config/loader plumbing** (`main.py`, `core/latent/loader.py`): add
   `--prediction-type {epsilon,v_prediction}` + `--min-snr-gamma` (default 0=off) to the
   train-diffusion parser (`main.py:1454-1485`); pass to `DiffusionTrainer`; write
   `prediction_type`, `min_snr_gamma` into pre-training config.json (`:560-584`);
   `load_diffusion_pipeline` must read `prediction_type` from config.json and pass it to
   `FastSamplingPipeline` (`loader.py:160-167` — currently silently defaults to epsilon).

Launch recipe (-007, RTX job env update only): existing TRAIN_ARGS +
`--prediction-type v_prediction --min-snr-gamma 5 --noise-schedule cosine`.
(-006 zero-centering canary PASSED: checkpoint landed, epoch-1 snapshot occupancy 0.646
vs -005's 0.70 — epoch-1 is near-noise for both; judge at matched epochs 3/5. Min-SNR is
the expected big convergence lever. -007 supersedes -006.)

---

## Stage 0 — plumbing refactors (no behavior change; unblock Stage B)

1. **`_finalize()` hook** on `Diffusion3DPipeline.sample` (`pipeline.py:199-201`): base impl
   = current `(s+1)/2` unscale (applied to final + intermediates). Single point every
   consumer funnels through (trainer snapshots, walks, sample-diffusion, loader).
2. **`clip_sample` passthrough**: `FastSamplingPipeline.__init__` (`pipeline.py:398-406`)
   → `DDIMScheduler` (defaults `clip_sample=True`, clamp at `noise_scheduler.py:328-329`).
   Pixel mode keeps True; latent mode passes False (the [-1,1] clamp would silently crush
   ~32% of unit-variance latent values every step).
3. **Loader arch threading** (`loader.py:130-138`): read `channel_mult`,
   `attention_resolutions`, `num_res_blocks` (+ Stage A's `prediction_type`) from
   config.json with current defaults preserved; write them in train_diffusion's config dump.
4. **Fix pre-existing `NameError`** in `train_diffusion`'s `run_summary.json` write
   (`main.py:621-622` references `collection_dir`/`occupancy_stats` that only exist in
   train_gan — starts firing once fast latent runs actually complete instead of timing out).

Gate: pixel `sample-diffusion` against an existing checkpoint behaves identically.

---

## Stage B — latent diffusion at 64³ (moves diffusion back to the L4 fleet)

**New files:** `core/models/autoencoder/{__init__.py, vae3d.py, codec.py}`,
`core/training/autoencoder_trainer.py`.
**Modified:** `main.py` (train-autoencoder subcommand; train-diffusion `--latent-autoencoder`),
`model_factory.py` (`create_autoencoder`), `diffusion_trainer.py` (codec path),
`pipeline.py` (`LatentFastSamplingPipeline`), `loader.py` (latent branch).

1. **VAE3D** (KL-VAE): 64³×1 → **16³×4** latent; two stride-2 down blocks + mid;
   **GroupNorm** (not BN — frozen-eval/bf16 safety); fully-conv decoder (no flattened
   Linear — Stage C requirement); [0,1] in, logits internal, **`decode()` always returns
   sigmoid probs** (threshold-0.5 convention preserved everywhere; no BCE `pos_weight` —
   it would shift the optimal threshold off 0.5). Loss: BCE(logits) + KL·w (w=1e-6
   default, flag). ~3-8M params, base 32ch. If IoU gate misses: add Dice term / widen —
   never pos_weight.
2. **`LatentCodec`** (codec.py): holds (frozen VAE, per-channel `latent_shift[4]`,
   `latent_scale[4]`); `encode(x01)→z_norm` (fp32, no_grad, outside autocast),
   `decode(z_norm)→probs01`. All normalization knowledge lives here only.
   Shift/scale computed **deterministically from an unshuffled 256-sample prefix** in
   `main.train_diffusion` BEFORE the config.json write (config.json is rewritten from args
   every resumed slice — values must be reproducible, since a pure function of frozen VAE
   weights + dataset prefix). Also stored in trainer checkpoints; assert equality on resume.
3. **AutoencoderTrainer(BaseTrainer)**: BCE+KL train_step (returns `'loss'` key so
   is_best works, `base_trainer.py:377-387`); `validate()` = held-out IoU@0.5 + occupancy
   error (deterministic last-5% split) → `val_*` in epoch_metrics.jsonl for free;
   `_after_epoch` saves `recon_epoch_NNN.pt` (4 fixed held-out orig/recon pairs, fp16) —
   renderable with existing tooling. Run dirs `autoencoder_*`, `--resume` via existing
   `_find_resume_checkpoint`. Flags: `--latent-channels 4 --kl-weight 1e-6
   --base-channels 32 --checkpoint-freq 1 --max-samples --resume`.
4. **DiffusionTrainer latent mode**: `DiffusionTrainer(codec=Optional[LatentCodec])` —
   in train_step, codec set ⇒ `x_0 = codec.encode(structure.unsqueeze(1))` replaces the
   `*2-1` line (encode BEFORE the `_x0_shape` assignment at :267 so fixed snapshot noise +
   slerp walk anchors are automatically latent-shaped); VAE is a trainer attribute, never a
   UNet submodule ⇒ EMA/optimizer/clip all stay correct (`:113-126`, `main.py:505-509`).
   `_snapshot_pipeline` builds `LatentFastSamplingPipeline` (codec `_finalize` override =
   decode; `clip_sample=False`) ⇒ snapshots + `walk_epoch_NNN.pt` contain **decoded [0,1]
   volumes** — all existing render tooling works unchanged.
5. **Latent UNet** (existing UNet3D, no changes): `void_dim=16, in/out=4,
   model_channels=128, channel_mult=[1,2,4], attention_resolutions=[8,4]` (not [1,2,4,8]
   — 2³×1024 bottleneck is wrong at void 16). Arch fields go in config.json (Stage 0.3).
6. **Run-dir prefix `latdiff_*`** (not `diffusion_*`) — the pixel chain's resume glob
   (`main.py:196-210`) must never cross-resume a latent run.
7. **Self-contained run dir**: copy `autoencoder.pt` (+ sha256 in config.json) into the
   latent run dir at first slice (idempotent) — survives container/region changes.
8. **Loader/CLI surface**: `load_diffusion_pipeline` latent branch (config has `latent`
   block ⇒ build codec, return latent pipeline) — `latent-walk --backend diffusion` works
   unchanged (noise shape already derives from config void_dim/num_channels,
   `main.py:1086-1087`). `sample-diffusion` trainer-ckpt branch already delegates to the
   loader — free. Reject `--sampler ddpm` and `--save-trajectory` in latent mode.

**Throughput expectation (honest)**: encoder compute is trivial; disk I/O dominates —
expect **3-8 min/epoch full 20k data on an L4** (vs 30 min for 2k pixel samples), 6-15
epochs/slice. Precomputed-latents caching is v2 only.

**v1 scope cuts**: no adversarial/perceptual VAE loss; posterior mean only; no latent
color mode; no v_prediction/conditioning/ddpm/trajectory in latent mode; no VAE EMA;
no precomputed latents; no 128³ decode exercised.

---

## Stage C — hi-res readiness

Constraint only: decoder fully convolutional (enforced in Stage B.1). A future 32³-latent
or 128³-VAE run needs no architecture surgery. No code now.

## GAN slab levers (parked — user decides from reference PNGs)

Either reweight `STOREY_WEIGHTS` (`shodhan.py:30-31`) + regenerate + fine-tune, or
zero-retrain GANSpace direction filter on the trained color GAN. Not in this plan's scope.

---

## Ops / sequencing

1. **Now**: Stage A + Stage 0 code + nano CPU tests → commit → Cloud Build cu128 → flip
   RTX job env to `RUN_ID=diff-shodhan-007` + new flags → canary (checkpoint + epoch-1/3/5
   snapshot occupancy vs -005/-006 baselines 0.70/0.646).
2. **Same session**: Stage B code + nano tests → commit → cu126 image build → launch
   `vae-shodhan-001` on the L4 `deepsculpt-train` job via env-override slices.
   **Gate: held-out IoU@0.5 ≥ 0.95 and |occupancy error| < 0.01.**
3. **After gate**: `latdiff-shodhan-001` on the L4 fleet; first canary with
   `--max-samples 2000`; then full-data slices. Retire the RTX chain once latent
   diffusion demonstrably trains (deadline kills it 2026-07-14 06:00 UTC regardless —
   do not extend it after latent works).
4. Any new scheduler goes under `deepsculpt-auto-shutdown`. All launches canary-verified
   by checkpoint-in-GCS.

## Verification

- **Nano (CPU, deepSculpt pyenv)**: Stage A — v-pred + min-SNR + cosine train_step on
  2×16³ dict batch (finite loss); DDIM + DPM v-pred sample round-trip in [0,1].
  Stage 0 — pixel sample() unchanged. Stage B — VAE fwd/bwd at void 16; encode/decode
  shape+range round-trip; codec train_step; decoded snapshot in [0,1]; latent std ≈ 1
  un-truncated (clip_sample test).
- **Cloud canaries**: -007 checkpoint + occupancy trend; VAE recon snapshots + IoU gate
  from epoch_metrics.jsonl; latdiff epoch-1 checkpoint + decoded voxel renders showing
  structure earlier than pixel baseline; measured latent epoch time logged.
- Existing test suite: `tests/test_latent_ops.py` stays green (only trusted suite).

## Risks (distilled)

| Risk | Mitigation |
|---|---|
| DDIM clip_sample crushes latents (silent) | `clip_sample=False` in latent pipeline + std≈1 unit test |
| latent shift/scale drift across resumed slices | deterministic prefix computation + checkpoint assert |
| loader rebuilds wrong UNet arch | config.json arch fields + loader threading (Stage 0) |
| VAE IoU gate missed | gate BEFORE diffusion spend; Dice/widen fallback |
| resume-glob cross-family | `latdiff_*` prefix |
| logits escaping VAE (threshold breaks) | decode() always sigmoids; rule enforced in codec |
| min-SNR formula wrong per pred-type | explicit branch + nano numeric test |

## Docs to update (same commits)

- **CLAUDE.md**: new train-diffusion flags + -007 recipe; `train-autoencoder` + VAE module;
  latent mode + latdiff recipe; diffusion returns to L4 (RTX tier retired); config.json
  arch/latent fields note.
- **Spec doc**: write approved design to
  `docs/superpowers/specs/2026-07-13-diffusion-speed-latent-design.md` (brainstorming flow)
  and commit. This plan doubles as the implementation plan (plan-mode replaces the
  separate writing-plans step).
- No README/DOCS.md impact.
