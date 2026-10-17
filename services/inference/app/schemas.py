from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class InferenceRequest(BaseModel):
    input_vector: Optional[List[float]] = Field(default=None, max_length=4096, description="Optional noise vector")
    num_samples: int = Field(default=1, ge=1, le=8)
    return_visualization: bool = Field(default=False)
    visualization_format: str = Field(default="png", description="png|gif|obj|stl")
    latent_walk: bool = Field(default=False, description="Generate a latent walk animation")
    latent_steps: int = Field(default=12, ge=2, le=48)
    num_inference_steps: int = Field(default=50, ge=5, le=250)
    guidance_scale: float = Field(default=1.0, ge=0.0, le=20.0)
    conditioning: Optional[List[float]] = Field(default=None, max_length=4096, description="Conditioning vector for conditional diffusion")
    class_label: Optional[int] = Field(default=None, ge=0)
    conditioning_mode: Optional[str] = Field(
        default=None,
        description="vector|one_hot|class_index (only for conditional diffusion)",
    )


class InferenceResponse(BaseModel):
    request_id: str
    result: Dict[str, Any]
    model_run_id: Optional[str]
    artifact_uri: Optional[str]
    visualization_uri: Optional[str] = None


class TrainRequest(BaseModel):
    params: Dict[str, Any] = Field(default_factory=dict)


class TrainResponse(BaseModel):
    status: str
    execution_id: Optional[str]
    submitted_params: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    model_run_id: Optional[str]
    artifact_uri: Optional[str]
    loaded_at: Optional[str]


class DatasetVisualizationRequest(BaseModel):
    gcs_uri: str = Field(description="GCS URI to a .npy volume")
    visualization_format: str = Field(default="png", description="png|gif|obj|stl")


class ModelPointerResponse(BaseModel):
    request_id: str
    pointer: Dict[str, Any]


class TrainStatusResponse(BaseModel):
    request_id: str
    execution_id: str
    status: str
    raw: Dict[str, Any]


class MlflowLastRunResponse(BaseModel):
    request_id: str
    run_id: Optional[str]
    status: Optional[str]
    metrics: Dict[str, Any]
    params: Dict[str, Any]
    tags: Dict[str, Any]
