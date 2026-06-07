# DeepSculpt Architecture

> **Post-2026-06-07 GCP migration:** the canonical training path is **RunPod-driven** (`runpod/` + `.github/workflows/`), with artifacts in `gs://garassino-ml-artifacts/deepsculpt/` (region `europe-west1`, project `garassino-ml`). See [`runpod.md`](./runpod.md) and [`gcs_layout.md`](./gcs_layout.md). The Cloud Run inference/MLflow services described below remain valid for local docker-compose dev and on-demand inference, but **no GCP compute is always-on** — training is show-and-destroy. Pod auth uses Workload Identity Federation via `garassino-op`'s `gh-actions` pool; **no service-account JSON keys anywhere**.

## Overview (local + on-demand compute layout)
This repo also ships a three-service shape for local dev and on-demand inference:
1. **GPU Training Job (Cloud Run Job)**: runs on-demand training, logs to MLflow, uploads model artifacts to GCS, updates a latest model pointer.
2. **GPU Inference API (Cloud Run Service)**: FastAPI service for `/infer`, `/train` trigger, and model reload. Loads the latest model from GCS on startup.
3. **CPU MLflow Tracking Server (Cloud Run Service)**: stateless MLflow server backed by Cloud SQL Postgres and GCS artifact store.

## Interaction Flow
- Inference service reads `LATEST_MODEL_POINTER_PATH` from GCS, downloads model, and serves inference.
- `/train` on inference triggers the Cloud Run Job for training via the Cloud Run Jobs API.
- Training job logs to MLflow, uploads model to GCS, and updates `latest.json`.

## Required Environment Variables
### Common
- `MODELS_BUCKET`: GCS bucket or bucket root for model artifacts (e.g., `my-bucket`)
- `LATEST_MODEL_POINTER_PATH`: pointer JSON location (default: `gs://<bucket>/models/latest.json`)

### Inference Service
- `PROJECT_ID`
- `REGION`
- `TRAIN_JOB_NAME`
- `ALLOW_GCLOUD_FALLBACK` (optional, default `false`)

### Training Job
- `MLFLOW_TRACKING_URI`
- `MODELS_BUCKET`
- `LATEST_MODEL_POINTER_PATH` (optional)
- `TRAIN_REQUEST_JSON` (optional)
- `TRAINING_MODE` (`gan` or `diffusion`, default: `diffusion`)
- `TRAIN_LR`, `TRAIN_EPOCHS`, `TRAIN_SEED`, `NOISE_DIM`, `VOID_DIM`, `TRAIN_BATCH_SIZE`, `MODEL_TYPE`, `COLOR_MODE`, `MODEL_VARIANT` (optional, overrides)
- `DIFFUSION_MODEL_TYPE`, `DIFFUSION_SCHEDULE`, `DIFFUSION_TIMESTEPS`, `DIFFUSION_PREDICTION_TYPE` (diffusion only)

### MLflow Server
- `BACKEND_STORE_URI` (Postgres)
- `ARTIFACT_ROOT` (GCS prefix)

## Local Development
### MLflow (dev only)
```
MLFLOW_IMAGE=ghcr.io/mlflow/mlflow:v2.13.1 docker compose up -d mlflow
```

### Inference (CPU fallback)
```
export MODELS_BUCKET=my-bucket
export LATEST_MODEL_POINTER_PATH=gs://my-bucket/models/latest.json
uvicorn services.inference.app.main:app --host 0.0.0.0 --port 8080
```

### Training (CPU fallback)
```
export MLFLOW_TRACKING_URI=http://localhost:8080
export MODELS_BUCKET=my-bucket
python -m services.trainer.train_entrypoint
```

## Sample Requests
### Trigger training
```
curl -X POST http://localhost:8080/train \
  -H "Content-Type: application/json" \
  -d '{"params": {"training_mode": "gan", "model_variant": "skip_mono", "epochs": 5}}'
```

### Trigger diffusion training
```
curl -X POST http://localhost:8080/train \
  -H "Content-Type: application/json" \
  -d '{"params": {"training_mode": "diffusion", "color_mode": 1, "diffusion_model_type": "unet3d", "diffusion_schedule": "cosine", "diffusion_timesteps": 1000}}'
```

### Run inference with visualization
```
curl -X POST http://localhost:8080/infer \
  -H "Content-Type: application/json" \
  -d '{"num_samples": 1, "return_visualization": true, "visualization_format": "png"}'
```

### Run inference with latent walk GIF
```
curl -X POST http://localhost:8080/infer \
  -H "Content-Type: application/json" \
  -d '{"return_visualization": true, "visualization_format": "gif", "latent_walk": true, "latent_steps": 12}'
```

### Run inference with mesh export
```
curl -X POST http://localhost:8080/infer \
  -H "Content-Type: application/json" \
  -d '{"num_samples": 1, "return_visualization": true, "visualization_format": "obj", "num_inference_steps": 50}'
```

### Visualize a dataset volume from GCS
```
curl -X POST http://localhost:8080/visualize-dataset \
  -H "Content-Type: application/json" \
  -d '{"gcs_uri": "gs://my-bucket/data/sample.npy", "visualization_format": "gif"}'
```

### Conditional diffusion inference
```
curl -X POST http://localhost:8080/infer \
  -H "Content-Type: application/json" \
  -d '{"num_samples": 1, "num_inference_steps": 50, "conditioning": [0.0, 0.1, 0.2], "class_label": 0}'
```

### Check latest model pointer
```
curl http://localhost:8080/models/latest
```

### Check training execution status
```
curl http://localhost:8080/train/status/projects/<PROJECT_ID>/locations/<REGION>/jobs/<JOB_NAME>/executions/<EXECUTION_ID>
```

### Check last MLflow run
```
curl http://localhost:8080/mlflow/last-run
```
## Model Variants

### GAN variants (MODEL_VARIANT)
- `skip_mono`: Skip‑connection generator, monochrome output (best default for speed/quality).
- `skip_color`: Skip‑connection generator, multi‑channel color output (best GAN quality).
- `complex_mono`: Deeper generator with internal skip connections, monochrome output.
- `complex_color`: Deeper generator with internal skip connections, color output.

### Diffusion defaults
- `diffusion_model_type=unet3d`: 3D U‑Net with time embedding.
- `diffusion_schedule=cosine`: typically best stability/quality.
- `diffusion_prediction_type=epsilon`: standard noise‑prediction objective.
- `diffusion_conditioning_mode=vector|one_hot|class_index`: conditioning schema.

## Docs Index
1. `docs/architecture.md`
2. `docs/training.md`
3. `docs/inference.md`
4. `docs/operations.md`
5. `docs/runpod.md` — RunPod + GCS deploy (canonical training path)
6. `docs/gcs_layout.md` — bucket structure + run_id convention
