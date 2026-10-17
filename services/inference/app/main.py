import json
import logging
import os
import uuid
from typing import Any, Dict

import numpy as np
from fastapi import FastAPI, HTTPException
import requests

from services.inference.app.gcp_run_job import trigger_training_job
from services.inference.app.model_loader import (
    get_cached_model,
    load_latest_model,
    run_latent_walk,
    run_inference,
)
from services.inference.app.schemas import (
    DatasetVisualizationRequest,
    HealthResponse,
    InferenceRequest,
    InferenceResponse,
    MlflowLastRunResponse,
    ModelPointerResponse,
    TrainRequest,
    TrainResponse,
    TrainStatusResponse,
)
from services.inference.app.settings import get_settings
from services.inference.app.uri_utils import normalize_execution_name, parse_allowed_gcs_uri
from services.inference.app.visualization import (
    make_and_upload_gif,
    make_and_upload_gif_from_slices,
    make_and_upload_mesh,
    make_and_upload_png,
    save_gif_from_volume_slices,
    upload_visualization,
)
from google.cloud import storage


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("inference")

app = FastAPI(title="DeepSculpt Inference API")


@app.on_event("startup")
def startup() -> None:
    settings = get_settings()
    settings.validate_for_inference()
    try:
        load_latest_model(settings.latest_model_pointer_path)
        logger.info("Loaded latest model on startup")
    except Exception as exc:
        logger.error("Failed to load model on startup: %s", exc)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    bundle = get_cached_model()
    return HealthResponse(
        status="ok",
        model_run_id=bundle.run_id if bundle else None,
        artifact_uri=bundle.artifact_uri if bundle else None,
        loaded_at=bundle.loaded_at if bundle else None,
    )


@app.post("/reload-model", response_model=HealthResponse)
def reload_model() -> HealthResponse:
    settings = get_settings()
    settings.validate_for_inference()
    bundle = load_latest_model(settings.latest_model_pointer_path)
    return HealthResponse(
        status="reloaded",
        model_run_id=bundle.run_id,
        artifact_uri=bundle.artifact_uri,
        loaded_at=bundle.loaded_at,
    )


@app.post("/infer", response_model=InferenceResponse)
def infer(request: InferenceRequest) -> InferenceResponse:
    request_id = str(uuid.uuid4())
    settings = get_settings()
    settings.validate_for_inference()

    bundle = get_cached_model()
    if not bundle:
        raise HTTPException(status_code=503, detail="Model not loaded")

    logger.info("infer request_id=%s model_run_id=%s", request_id, bundle.run_id)

    input_vector = None
    if request.input_vector is not None:
        input_vector = np.array(request.input_vector, dtype=np.float32)

    try:
        conditioning_mode = request.conditioning_mode
        if conditioning_mode:
            conditioning_mode = conditioning_mode.lower()
            if conditioning_mode not in {"vector", "one_hot", "class_index"}:
                raise HTTPException(status_code=400, detail="Invalid conditioning_mode")
            if conditioning_mode == "class_index" and request.class_label is None:
                raise HTTPException(status_code=400, detail="class_label required for class_index")
            if conditioning_mode in {"vector", "one_hot"} and request.conditioning is None:
                if request.class_label is None:
                    raise HTTPException(status_code=400, detail="conditioning or class_label required")
                # Auto-convert class_label to one-hot vector
                num_classes = bundle.diffusion_num_classes or bundle.diffusion_conditioning_dim
                one_hot = [0.0] * int(num_classes)
                idx = int(request.class_label) % int(num_classes)
                one_hot[idx] = 1.0
                request.conditioning = one_hot
        output = run_inference(
            bundle,
            input_vector,
            request.num_samples,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            conditioning=request.conditioning,
            class_label=request.class_label,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    result: Dict[str, Any] = {
        "shape": list(output.shape),
        "mean": float(output.mean()),
        "std": float(output.std()),
    }

    visualization_uri = None
    if request.return_visualization:
        fmt = request.visualization_format.lower()
        if fmt == "png":
            visualization_uri = make_and_upload_png(
                output,
                f"gs://{settings.models_bucket}/inference-visuals/{request_id}.png",
            )
        elif fmt == "gif":
            if request.latent_walk and bundle.training_mode != "diffusion":
                volumes = run_latent_walk(bundle, request.latent_steps)
                visualization_uri = make_and_upload_gif(
                    volumes,
                    f"gs://{settings.models_bucket}/inference-visuals/{request_id}.gif",
                )
            else:
                visualization_uri = make_and_upload_gif_from_slices(
                    output,
                    f"gs://{settings.models_bucket}/inference-visuals/{request_id}.gif",
                )
        elif fmt in {"obj", "stl"}:
            visualization_uri = make_and_upload_mesh(
                output,
                f"gs://{settings.models_bucket}/inference-visuals/{request_id}.{fmt}",
                fmt,
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported visualization_format: {fmt}")

    return InferenceResponse(
        request_id=request_id,
        result=result,
        model_run_id=bundle.run_id,
        artifact_uri=bundle.artifact_uri,
        visualization_uri=visualization_uri,
    )


@app.post("/train", response_model=TrainResponse)
def train(request: TrainRequest) -> TrainResponse:
    request_id = str(uuid.uuid4())
    settings = get_settings()
    settings.validate_for_train_trigger()

    logger.info("train request_id=%s params=%s", request_id, json.dumps(request.params))

    try:
        response = trigger_training_job(
            project_id=settings.project_id,
            region=settings.region,
            job_name=settings.train_job_name,
            train_params=request.params,
            allow_gcloud_fallback=settings.allow_gcloud_fallback,
        )
    except Exception as exc:
        logger.exception("train request_id=%s submit failed", request_id)
        raise HTTPException(status_code=500, detail="Failed to submit training job") from exc

    execution_id = response.get("name") or response.get("metadata", {}).get("name")

    return TrainResponse(
        status="submitted",
        execution_id=execution_id,
        submitted_params=request.params,
    )


@app.get("/models/latest", response_model=ModelPointerResponse)
def latest_model_pointer() -> ModelPointerResponse:
    request_id = str(uuid.uuid4())
    settings = get_settings()
    settings.validate_for_inference()

    bucket_name, blob_path = settings.latest_model_pointer_path.replace("gs://", "").split("/", 1)
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(blob_path)
    if not blob.exists():
        raise HTTPException(status_code=404, detail="latest model pointer not found")

    pointer = json.loads(blob.download_as_text())
    return ModelPointerResponse(request_id=request_id, pointer=pointer)


@app.get("/train/status/{execution_id:path}", response_model=TrainStatusResponse)
def train_status(execution_id: str) -> TrainStatusResponse:
    request_id = str(uuid.uuid4())
    settings = get_settings()
    settings.validate_for_train_trigger()

    try:
        execution_id = normalize_execution_name(
            execution_id, settings.project_id, settings.region, settings.train_job_name
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    token = None
    try:
        from google.auth.transport.requests import Request
        import google.auth

        credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        credentials.refresh(Request())
        token = credentials.token
    except Exception as exc:
        logger.exception("train_status request_id=%s token refresh failed", request_id)
        raise HTTPException(status_code=500, detail="Failed to get access token") from exc

    url = f"https://run.googleapis.com/v2/{execution_id}"
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, headers=headers, timeout=20)
    if response.status_code >= 400:
        logger.warning(
            "train_status request_id=%s run api %s: %s",
            request_id,
            response.status_code,
            response.text[:500],
        )
        raise HTTPException(status_code=response.status_code, detail="Run API request failed")
    data = response.json()
    status = data.get("status", "unknown")
    return TrainStatusResponse(request_id=request_id, execution_id=execution_id, status=status, raw=data)


@app.get("/mlflow/last-run", response_model=MlflowLastRunResponse)
def mlflow_last_run() -> MlflowLastRunResponse:
    request_id = str(uuid.uuid4())
    settings = get_settings()
    if not settings.mlflow_tracking_uri:
        raise HTTPException(status_code=400, detail="MLFLOW_TRACKING_URI not configured")

    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except Exception as exc:
        logger.exception("mlflow client import failed")
        raise HTTPException(status_code=500, detail="MLflow client unavailable") from exc

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient()
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT", "deepsculpt")
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return MlflowLastRunResponse(
            request_id=request_id,
            run_id=None,
            status=None,
            metrics={},
            params={},
            tags={},
        )

    runs = client.search_runs(
        [experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        return MlflowLastRunResponse(
            request_id=request_id,
            run_id=None,
            status=None,
            metrics={},
            params={},
            tags={},
        )

    run = runs[0]
    return MlflowLastRunResponse(
        request_id=request_id,
        run_id=run.info.run_id,
        status=run.info.status,
        metrics=run.data.metrics,
        params=run.data.params,
        tags=run.data.tags,
    )


@app.post("/visualize-dataset", response_model=InferenceResponse)
def visualize_dataset(request: DatasetVisualizationRequest) -> InferenceResponse:
    request_id = str(uuid.uuid4())
    settings = get_settings()
    settings.validate_for_inference()

    if not request.gcs_uri.endswith(".npy"):
        raise HTTPException(status_code=400, detail="gcs_uri must point to a .npy file")

    try:
        bucket_name, blob_path = parse_allowed_gcs_uri(request.gcs_uri, settings.models_bucket)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(blob_path)
    if not blob.exists():
        raise HTTPException(status_code=404, detail="Dataset object not found")

    local_path = f"/tmp/{request_id}.npy"
    blob.download_to_filename(local_path)
    volume = np.load(local_path, allow_pickle=False)

    if volume.ndim == 3:
        volume = volume[None, None, ...]
    elif volume.ndim == 4:
        volume = volume[None, ...]

    fmt = request.visualization_format.lower()
    visualization_uri = None
    if fmt == "png":
        visualization_uri = make_and_upload_png(
            volume,
            f"gs://{settings.models_bucket}/dataset-visuals/{request_id}.png",
        )
    elif fmt == "gif":
        local_gif = f"/tmp/{request_id}.gif"
        save_gif_from_volume_slices(volume, local_gif)
        visualization_uri = upload_visualization(
            local_gif,
            f"gs://{settings.models_bucket}/dataset-visuals/{request_id}.gif",
        )
    elif fmt in {"obj", "stl"}:
        visualization_uri = make_and_upload_mesh(
            volume,
            f"gs://{settings.models_bucket}/dataset-visuals/{request_id}.{fmt}",
            fmt,
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported visualization_format: {fmt}")

    try:
        os.remove(local_path)
    except OSError:
        pass
    if fmt == "gif":
        try:
            os.remove(local_gif)
        except OSError:
            pass

    return InferenceResponse(
        request_id=request_id,
        result={"source": request.gcs_uri},
        model_run_id=None,
        artifact_uri=None,
        visualization_uri=visualization_uri,
    )
