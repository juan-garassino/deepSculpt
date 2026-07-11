# DeepSculpt

> **GCP migration note (2026-07-05):** Lives in `garassino-ml`. Images on `ghcr.io/juan-garassino/deepsculpt-runpod` (public, canonical) + mirrored to `europe-west1-docker.pkg.dev/garassino-ml/ml-images/deepsculpt` (cloud jobs pull from GAR). Artifacts in `gs://garassino-ml-artifacts/deepsculpt/`. No always-on resources — **training runs on Cloud Run jobs** (Vertex kept, RunPod dormant). See root `CLAUDE.md` § "GCP architecture".

## What this project is
A 3D generative art system that learns to create sculptures from scratch.
It uses two types of AI models:
- A **GAN** (two networks fighting each other — one creates shapes, one judges them)
- A **Diffusion model** (gradually removes noise to reveal a shape, like developing a photo)

The sculptures are stored as 3D grids of numbers (numpy arrays):
- Monochrome shapes: a cube of numbers, each cell says "empty" (0) or "solid" (1)
- Color shapes: same cube but each cell also has a color encoded as 4 numbers (RGBA)

## What I'm trying to do
Train the AI to generate believable 3D shapes it has never seen before,
then navigate the "space" of all possible shapes it has learned — finding
new sculptures, blending between two shapes, or asking "show me something rounder".

## The pipeline (in order)
1. **Generate training data** — create thousands of example 3D shapes using math
2. **Validate the data** — make sure the shapes look right before training
3. **Train the GAN or diffusion model** on those shapes
4. **Evaluate** — check if the AI is learning or getting stuck
5. **Explore** — navigate the space of shapes the AI has learned

## Stack
- Python, NumPy, PyTorch
- Shapes stored as `.npy` files
- Training runs on GPU (CUDA)

## Skills available to help
- `ds-datagen` — help writing shape generation code
- `ds-dataval` — help writing data checking and visualization code
- `ds-gan` — help with GAN architecture and training code
- `ds-diffusion` — help with diffusion model code
- `ds-latent` — help with latent space navigation code
- `ds-improve` — autonomous agent that runs improvement loops on the codebase

## ⚡ AGENT BEHAVIOR — READ THIS FIRST

When the user says anything like:
- "this isn't working" / "it's broken" / "fix this"
- "the training is getting worse"
- "the shapes all look the same"
- "make it better" / "help me improve this"
- "something is wrong"

**Do NOT just give advice. Immediately launch the `ds-improve` agent.**
Run it autonomously: measure diversity and validity of generated shapes,
diagnose what's failing, apply targeted fixes to the code, and report
what changed.

When the user says anything like:
- "explain how X works"
- "I don't understand X"
- "what is X"

Use the skills interactively to explain — don't launch the agent.

When the user says anything like:
- "write the training loop"
- "help me build the generator"
- "implement X"

Use the relevant skill interactively to write the code together.

## How to ask (no jargon needed)
- "The shapes all look the same" → launches ds-improve (mode collapse)
- "Training is crashing" → launches ds-improve
- "Something looks wrong with the data" → launches ds-improve
- "Blend between two sculptures" → interactive, uses ds-latent
- "Explain what a GAN is" → interactive explanation
- "Write the shape generator" → interactive, uses ds-datagen

## Current status
- **PyTorch v2 is the only path.** TensorFlow/Keras legacy is archived under `boilerplate/`.
- **CLI works.** `python -m deepsculpt.main {train-gan,train-diffusion,generate-data,sample-gan,sample-diffusion,latent-walk,latent-traverse,latent-directions,preprocess,visualize,benchmark,evaluate,export}`.
- **Latent navigation is live** (`core/latent/`): slerp/lerp walks, per-dim traversal, GANSpace/PCA semantic directions, diffusion noise-space walks (`latent-walk --backend diffusion`), rendered as GIF/PNG/OBJ/STL via `core/visualization/volume_export.py`. Pass negative alpha lists as `--alphas="-2,0,2"`.
- **Optional-dep imports are guarded with `except Exception`** (Prefect 2/3 raises ValidationError, not ImportError — don't narrow these back).
- **Shodhan grammar v2 is the active data generator** (`--structure-preset shodhan`, generate-data default): Corbusier Dom-ino skeleton × Umemoto carved massing at 64³ — full-square slabs (the one template element), structured 3×3/3×4/4×4 column arrays, strip-cut intermediate slabs, column-to-column polychrome walls (2/level, some double-height), flush vertical-fin brise-soleil, wall-anchored pipes, carved blocks (1-voxel shells, slab-to-slab openings), roof terraces with pavilion block. Deterministic per seed (`generate_structure_with_params`); per-sample `params_*.json`; spec at `docs/superpowers/specs/2026-07-05-shodhan-dataset-design.md`. Gate before GPU spend: `python scripts/shodhan_probe.py --n 200` (masked-IoU + frequency gates + contact sheet + 3D HTML). Legacy presets kept: `architectural`, `generic`.
- **Test suite has pre-existing import bugs** (all `tests/unit/*.py` and `tests/integration/*` import the dead `deepSculpt` casing — package is `deepsculpt`). Don't trust `pytest` as a green light; smoke-test via the CLI instead. `tests/test_latent_ops.py` is the exception (pure tensor tests; `tests/conftest.py` is torch-only now, coverage is opt-in).
- **Cloud training runs on Cloud Run jobs** (user pivot 2026-07-03, mirrors the `~/Desktop/garassino-ml` house pattern): job `deepsculpt-train` in `garassino-ml`/`europe-west1` — **L4 GPU** (`--gpu 1 --gpu-type nvidia-l4 --no-gpu-zonal-redundancy`, 8 CPU/32Gi, `--max-retries 0`; quota allows 2 concurrent), same container + entrypoint as Vertex/RunPod, ADC auth via the default compute SA. **GPU jobs are hard-capped at `--task-timeout 3600`** → long training = chained executions with `--resume` in `TRAIN_ARGS` (continues the latest run dir from its newest `checkpoint_epoch_N.pth`; pass `--checkpoint-freq 1-2` so a timed-out slice loses ≤1 epoch). Launch: `gcloud run jobs execute deepsculpt-train --project garassino-ml --region europe-west1` (override per-run env with `--update-env-vars "^@^K=V@K2=V2"`). Pin the image to a `:$(git sha)` GAR tag to test branch builds before merging.
- **Unattended slice chaining via Cloud Scheduler** (2026-07-11): `deepsculpt-gan-chain-a/b` (90-min cadence, baked template env = mono gan-shodhan-001), `deepsculpt-diff-chain-a` + `deepsculpt-color-chain` (3h each, offset, env overrides in the POST body) keep training without a laptop. OAuth SA `deepsculpt-runpod-runtime@` (run.invoker). **Pause/delete these when quality target reached — they fire forever (~$30/day).**
- **Remote learning signal**: trainer INFO logs never reach Cloud Logging (root logger is CLI-configured; `training.log` files stay empty) — read `logs/epoch_metrics.jsonl` and `snapshots/epoch_NNN.{json,pt}` from GCS instead; render with `scripts/render_run_snapshots.py <run_id>`. GAN snapshots come from the **EMA generator, which looks diffuse/blurry mid-training — judge quality on the RAW generator** (pull a checkpoint, sample `generator_state_dict`).
- **Per-epoch fixed-vector snapshots + walk dumps (both trainers)**: `fixed_noise` is CPU-seeded (1234) so the same latents persist across chained slices. GAN saves `snapshots/walk_epoch_NNN.pt` every epoch (16-step slerp between fixed anchors through the **raw** generator, fp16). Diffusion saves fixed-noise `epoch_NNN.pt` samples per `snapshot_freq` epoch (4 × 25-step DDIM) + `walk_epoch_NNN.pt` every 5th. `config.json` is written **before** training (timed-out slices used to never get one), and `load_diffusion_pipeline` accepts trainer `checkpoint_epoch_N.pth` (EMA weights + rebuilt scheduler + sibling config.json) — `diffusion_final.pt` no longer required for walks.
- **Walk beauty pass**: `python scripts/render_walk.py <walk_volumes.pt|walk_epoch_NNN.pt> [--orbit 90 --boomerang --color storm]` — marching-cubes + depth-shaded trisurf GIF (ax.voxels is unusably slow at 64³); iterate aesthetics from saved volumes without re-running the generator.
- **GAN color mode (`train-gan --color`)**: one 13-channel semantic-class field (ch 0 = empty, 1-12 = shodhan element classes). Skip generator grows a 13ch softmax head, spectral_norm disc takes 13ch input (other archs untouched); reals = one-hot of the colors tensor with 0.95/0.05 label smoothing; occupancy penalty uses non-empty prob mass (soft) / argmax>0 fraction (hard). Snapshots, walk dumps, and latent-walk exports save **argmax int8 class volumes**; `scripts/render_walk.py` auto-detects integer volumes and renders per-class isosurfaces with the element palette. Mono path is bit-identical — everything gates on color mode. Caveat: shodhan `generate-data` is hardcoded 64³ (`--void-dim` is ignored by that preset).
- **Working GAN recipe (gan-cr-005)**: `--model-type skip --discriminator-type spectral_norm --gan-loss-type softplus --ttur-ratio 1.0 --batch-size 16 --mixed-precision` (bf16 autocast). Hard-won: fp16 overflows critic logits (NaN); WGAN-GP + light disc diverges (logit race); the EMA-disc-for-gen-loss design was the root cause of gen-loss explosions (fixed: G trains vs live D).
- **Diffusion on the L4 needs all three**: `--model-channels 64 --grad-checkpoint --batch-size 16` + cudnn benchmark off (hardcoded for train-diffusion) — the 128ch UNet, hand-rolled attention (pre-SDPA), and cuDNN autotune workspaces each independently OOM'd the 22 GiB card.
- **Vertex AI path kept** (2026-06-12): `gh workflow run deploy-vertex.yml -f mode=train -f machine='n1-standard-8 / 1x T4' ...`; on Vertex the job runs AS the runtime SA (ADC). **T4 is the only usable GPU in europe-west1 Vertex training** (L4/G2 machine rejected; A100 quota 0). **torch must stay on cu12x wheels** (`--index-url .../whl/cu126` in Dockerfile) — Vertex T4 / Cloud Run L4 drivers are CUDA 12.x, cu130 silently falls back to CPU; the entrypoint logs GPU diagnostics on start, check `cuda available: True` in the first log page of every paid run. GCS data cache is keyed `data/void<dim>/`. `runpod-ctl.yml` lists/stops/terminates RunPod pods (no local API key).
- **MLflow tracking (partially deployed 2026-07-05)**: global server for the whole `garassino-ml` workspace, authored at `~/Desktop/garassino-ml/mlops/mlflow-server/` — image built to `europe-west1-docker.pkg.dev/garassino-ml/mlflow/mlflow:2.22.5`, SA `mlflow-server@` + bucket IAM done; **deploy blocked on user creating the Neon DB → secret `neon-mlflow-dsn`**. Client wiring is live (commit `1abe77f`): pass `MLFLOW_TRACKING_URI` env + `--mlflow` in TRAIN_ARGS → experiment `deepsculpt`, run name = `RUN_ID`, per-step metrics stream from the trainers. Artifacts land under `gs://garassino-ml-artifacts/mlflow/`.
- **This dev machine is old — no local training.** Verify with forward passes and nano runs only (void_dim 16, 1-2 minibatches max); real training goes to Cloud Run.

## Cloud training (RunPod + GCS + Claude-in-the-loop)
The `runpod/` directory contains the full deploy. Five modes:
- `MODE=train` — pure training, no Claude in the loop.
- `MODE=render` — one-off cloud inference (RENDER_CMD=latent-walk/sample-gan/... + RENDER_ARGS) against RUN_ID's checkpoints, HQ-GIF pass via `scripts/render_walk.py` (HQ_ARGS), skips the data pull; artifacts sync to `results/<RUN_ID>/`. Local machine only downloads.
- `MODE=research` — one-shot Claude reading `runpod/prompts/research.md`, runs experiments to `TIME_BUDGET`.
- `MODE=improve` — one-shot Claude reading `runpod/prompts/improve.md`, drives the `ds-improve` skill once.
- `MODE=self-improve` — **continuous** Claude loop; each iteration uses `runpod/prompts/self_improve.md` which references `runpod/prompts/autoresearch_program.md` (mirrored from the [020-autoresearch](../020-autoresearch/) reference repo). **Toggleable on/off** via a GCS-synced object (`gs://garassino-ml-artifacts/deepsculpt/control/self_improve.enabled`); `make toggle-on` / `make toggle-off` from the `runpod/` Makefile — no pod restart needed.

Pod auth: **GHA-minted short-lived bearer token (WIF through `garassino-op`'s `gh-actions` pool). No SA JSON anywhere.** Refreshed every 50 min by `.github/workflows/refresh-token.yml`. Canonical deploy path: `gh workflow run deploy-runpod.yml -f mode=...` (or `make deploy MODE=...`). Infra is show-and-destroy via `infra/gcp/` Terraform.

GCS layout: `gs://garassino-ml-artifacts/deepsculpt/{data,checkpoints/<run_id>,results/<run_id>,prompts-archive,control,state}/` (bucket region: **`europe-west1`**). Crash-safe: periodic background `gsutil rsync` + final sync on `EXIT`. Image: `ghcr.io/juan-garassino/deepsculpt-runpod:latest` (built/pushed by `.github/workflows/build-push.yml` on every master push). See `runpod/README.md`, `docs/runpod.md`, and `infra/gcp/README.md`.

## File structure
```
deepsculpt/
├── main.py                       # CLI entry — 13 subcommands
├── config.yaml                   # central hyperparameters
├── core/
│   ├── data/{generation,loaders,sparse,transforms}/   # shape gen, dataloaders, encoding
│   ├── latent/{ops,loader,directions}.py              # latent navigation: walks, traversal, PCA directions
│   ├── models/
│   │   ├── gan/{generator,discriminator}.py           # 5 gens, 8 discs incl. SelfAttention3D, LightDiscriminator
│   │   ├── diffusion/{unet,noise_scheduler,pipeline,pytorch_diffusion}.py  # pytorch_diffusion is unused reference code
│   │   ├── base_models.py, model_factory.py
│   ├── training/{gan_trainer,diffusion_trainer,base_trainer,optimizers,schedulers}.py
│   ├── utils/{logger,pytorch_utils,monitoring,performance_optimizer}.py
│   ├── visualization/{pytorch_visualization,volume_export}.py  # volume_export: pure GIF/PNG/mesh writers (no GCS)
│   └── workflow/{pytorch_workflow,pytorch_mlflow_tracking}.py
services/                         # Cloud Run trio: trainer job, inference FastAPI (has its own latent_walk endpoint), mlflow
notebooks/                        # 9 Jupyter notebooks (Colab + local)
scripts/                          # colab_train.py, colab_train_diffusion.py, autoresearch_report.py, preview_sample.py
tests/                            # pytest suite (most fixed via sed; 11 tier-2 errors remain — see Current status)
runpod/                           # RunPod + GCS deploy (Dockerfile, entrypoint.sh, Makefile, prompts/, scripts/)
infra/gcp/                        # show-and-destroy Terraform: runtime SA + WIF impersonation binding
.github/workflows/                # build-push (GHCR+GAR), deploy-vertex, deploy-runpod, runpod-ctl, refresh-token (cron), notify-telegram (cron)
docs/                             # architecture, training, operations, inference, runpod, gcs_layout
boilerplate/                      # archived TF/Keras v1 code (do not propagate)
checkpoints/                      # local checkpoint dir; legacy samples committed under data/11/
```
