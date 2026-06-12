# DeepSculpt

DeepSculpt is a 3‑service, GPU‑ready system for 3D voxel generation with GANs and diffusion models. It includes:
1. GPU training job (Cloud Run Job)
2. GPU inference API (FastAPI)
3. CPU MLflow tracking server (stateless)

## Architecture

1. **Trainer** runs on demand, logs to MLflow, uploads artifacts to GCS, and updates a latest model pointer.
2. **Inference** loads the latest model from GCS and serves `/infer`, `/train`, and visualization endpoints.
3. **MLflow** is stateless, backed by Cloud SQL Postgres and GCS.

See `docs/architecture.md` for details.

## Quick Start (Local)

1. Start services.
```bash
make docker-build
make docker-up
```

2. Trigger training.
```bash
curl -X POST http://localhost:8081/train \
  -H "Content-Type: application/json" \
  -d '{"params": {"training_mode": "gan", "model_variant": "skip_color", "epochs": 2}}'
```

3. Run inference with visualization.
```bash
curl -X POST http://localhost:8081/infer \
  -H "Content-Type: application/json" \
  -d '{"num_samples": 1, "return_visualization": true, "visualization_format": "png"}'
```

## API Summary

1. `GET /health`
2. `POST /infer`
3. `POST /train`
4. `POST /reload-model`
5. `POST /visualize-dataset`
6. `GET /models/latest`
7. `GET /train/status/{execution_id}`
8. `GET /mlflow/last-run`

## Latent-space navigation (CLI)

Explore a trained model's latent space directly from the CLI (CPU-friendly; the checkpoint's `config.json` must sit beside the weights):

```bash
# Interpolation walk between two seeds (GIF + meshes)
python -m deepsculpt.main --cpu latent-walk --checkpoint <run_dir>/ema_generator_final.pt \
  --seeds 12,77 --steps 30 --interp slerp --format gif --format obj

# Vary one z dimension at a time (contact-sheet PNG)
python -m deepsculpt.main --cpu latent-traverse --checkpoint <ckpt> --dims 0,3,9 --steps 9 --format png

# Discover semantic directions (GANSpace PCA) and render one
python -m deepsculpt.main --cpu latent-directions --checkpoint <ckpt> \
  --method ganspace --components 10 --apply 0 --alphas="-3,-1.5,0,1.5,3"

# Walk diffusion initial-noise space (deterministic DDIM per frame)
python -m deepsculpt.main --cpu latent-walk --backend diffusion \
  --checkpoint <run_dir>/diffusion_final.pt --seeds 5,9 --steps 12 --diffusion-steps 25
```

Pull trained weights from a RunPod run first: `gsutil -m rsync -r gs://garassino-ml-artifacts/deepsculpt/checkpoints/<run_id> ./checkpoints/<run_id>`.

## Training Modes

### GAN
Variants are mapped via `MODEL_VARIANT`:
1. `skip_mono`
2. `skip_color`
3. `complex_mono`
4. `complex_color`

Best default: `skip_color`.

### Diffusion
Defaults are:
1. `diffusion_model_type=unet3d`
2. `diffusion_schedule=cosine`
3. `diffusion_prediction_type=epsilon`

Best default: `unet3d` + `cosine` + `epsilon`, with `color_mode=1`.

## Visualizations

Inference can return:
1. `png` mid‑slice
2. `gif` latent walk or slice animation
3. `obj` mesh
4. `stl` mesh

Dataset visualizations are supported via `/visualize-dataset` for `.npy` volumes in GCS.

## Environment Variables

### Inference
1. `PROJECT_ID`
2. `REGION`
3. `TRAIN_JOB_NAME`
4. `MODELS_BUCKET`
5. `LATEST_MODEL_POINTER_PATH`

### Trainer
1. `MLFLOW_TRACKING_URI`
2. `MODELS_BUCKET`
3. `LATEST_MODEL_POINTER_PATH`
4. `TRAINING_MODE` (`gan` or `diffusion`)

### MLflow
1. `BACKEND_STORE_URI`
2. `ARTIFACT_ROOT`

## Train on RunPod (GPU + GCS + Claude-in-the-loop)

The `runpod/` directory ships a CUDA 12.8 + Claude Code container that runs on RunPod and syncs checkpoints/results to GCS. The image is built + pushed automatically by `.github/workflows/build-push.yml` on every push to `master`; deployment is a `gh workflow run` away — no local Docker required.

```bash
# Deploy via CI (canonical)
gh workflow run deploy-runpod.yml -f mode=research -f time_budget=3600 -f gpu_type='NVIDIA A100 80GB PCIe'
# or
make -C runpod deploy MODE=research TIME_BUDGET=3600
```

If you prefer the manual UI route, on RunPod create a GPU Pod with the image and set env vars (note: you must supply `GCS_ACCESS_TOKEN` yourself via `gcloud auth print-access-token`):

| Var | Notes |
|---|---|
| `ANTHROPIC_API_KEY` | required for `MODE=research` / `improve` / `self-improve`. GHA repo secret. |
| `GCS_BUCKET` | `garassino-ml-artifacts` (region `europe-west1`) |
| `GCS_PROJECT` | `garassino-ml` |
| `GCS_ACCESS_TOKEN` | short-lived OAuth2 token minted by GHA via WIF. No SA JSON. |
| `MODE` | `train` \| `research` \| `improve` \| `self-improve` (continuous + toggleable; see [`docs/runpod.md`](docs/runpod.md)) |
| `RUN_ID` | run identifier (used in GCS paths) |
| `TIME_BUDGET` | seconds (research/improve modes) |

Full deploy guide: [`docs/runpod.md`](docs/runpod.md). GCS layout: [`docs/gcs_layout.md`](docs/gcs_layout.md). Container reference: [`runpod/README.md`](runpod/README.md).

### GCS layout

```
gs://garassino-ml-artifacts/deepsculpt/
├── data/<dataset_name>/...
├── checkpoints/<RUN_ID>/{generator.pt, discriminator.pt, optimizer.pt, config.yaml}
├── results/<RUN_ID>/{experiments.tsv, claude.log, samples/, summary.md}
└── prompts-archive/<timestamp>-<mode>.md
```

The entrypoint pulls existing state on startup (resume support), rsyncs every 10 minutes during the run, and does a final push on `EXIT`.

## Docs

1. `docs/architecture.md`
2. `docs/training.md`
3. `docs/inference.md`
4. `docs/operations.md`
5. `docs/runpod.md` — RunPod + GCS deploy
6. `docs/gcs_layout.md` — GCS bucket structure and run_id convention
