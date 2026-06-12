# Inference

The inference service loads the latest model from GCS and serves a FastAPI API.

> Latent-space navigation (walks, traversal, semantic directions) lives in the core CLI — see "Latent-space navigation" in the root README. The service's `latent_walk` request option is a separate lerp-only implementation and keeps working unchanged.

## Endpoints

1. `GET /health`
2. `POST /infer`
3. `POST /reload-model`
4. `POST /visualize-dataset`
5. `GET /models/latest`
6. `GET /train/status/{execution_id}`
7. `GET /mlflow/last-run`

## Example: Basic inference
```bash
curl -X POST http://localhost:8081/infer \
  -H "Content-Type: application/json" \
  -d '{"num_samples": 1}'
```

## Example: PNG visualization
```bash
curl -X POST http://localhost:8081/infer \
  -H "Content-Type: application/json" \
  -d '{"num_samples": 1, "return_visualization": true, "visualization_format": "png"}'
```

## Example: Latent walk GIF
```bash
curl -X POST http://localhost:8081/infer \
  -H "Content-Type: application/json" \
  -d '{"return_visualization": true, "visualization_format": "gif", "latent_walk": true, "latent_steps": 12}'
```

## Example: Conditional diffusion
```bash
curl -X POST http://localhost:8081/infer \
  -H "Content-Type: application/json" \
  -d '{"num_samples": 1, "conditioning": [0.0, 0.1, 0.2], "class_label": 0, "guidance_scale": 1.5, "num_inference_steps": 50}'
```
