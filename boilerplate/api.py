"""
FastAPI server for DeepSculpt model inference.

This module provides a comprehensive API for DeepSculpt:
1. 3D model generation with various parameter controls
2. Model management and exploration features
3. Batch processing capabilities for multiple generations
4. Interactive visualization options
5. Real-time parameter tuning and feedback

To run the server:
    uvicorn api:app --reload
"""

import os
import io
import base64
import uuid
import json
import time
import tempfile
import numpy as np
import tensorflow as tf
from typing import List, Dict, Optional, Union, Any, Set
from datetime import datetime
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import shutil

from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile, Form, Query, Depends
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, validator, root_validator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DeepSculpt.API")

# Try to import DeepSculpt modules
try:
    # Import DeepSculpt modules
    from models import ModelFactory
    from workflow import Manager
    from visualization import Visualizer
    
    MODULES_AVAILABLE = True
    logger.info("DeepSculpt modules loaded successfully")
except ImportError as e:
    logger.error(f"Failed to import DeepSculpt modules: {e}")
    MODULES_AVAILABLE = False

# Create FastAPI app
app = FastAPI(
    title="DeepSculpt API",
    description="API for generating 3D models using DeepSculpt",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in dev, restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
API_KEYS = set(os.environ.get("API_KEYS", "test-key").split(","))
MODEL_CACHE = {}  # Cache for loaded models
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "./uploads")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./outputs")
HISTORY_DIR = os.environ.get("HISTORY_DIR", "./history")
MAX_HISTORY_ENTRIES = int(os.environ.get("MAX_HISTORY_ENTRIES", "100"))

# Create directories
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)

# Mount static files directory for serving images
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# Security
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_api_key(api_key: str = Depends(api_key_header)):
    if not API_KEYS or "test-key" in API_KEYS:
        # If no real API keys are set, skip validation
        return None
    
    if api_key not in API_KEYS:
        raise HTTPException(
            status_code=401,
            detail="Invalid API Key"
        )
    return api_key


# Pydantic models for request/response validation
class GenerationRequest(BaseModel):
    model_type: str = Field("skip", description="Type of model to use")
    noise_dim: int = Field(100, description="Dimension of noise vector")
    num_samples: int = Field(1, description="Number of samples to generate", ge=1, le=16)
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    slice_axis: int = Field(0, description="Axis for 2D slice visualization", ge=0, le=2)
    slice_position: float = Field(
        0.5, description="Position for slice (0.0-1.0)", ge=0.0, le=1.0
    )
    batch_id: Optional[str] = Field(None, description="Batch ID for related generations")
    save_3d: bool = Field(False, description="Whether to save the full 3D model data")
    visual_style: str = Field("default", description="Visual style for the output")
    
    class Config:
        schema_extra = {
            "example": {
                "model_type": "skip",
                "noise_dim": 100,
                "num_samples": 4,
                "seed": 42,
                "slice_axis": 0,
                "slice_position": 0.5,
                "save_3d": True,
                "visual_style": "default"
            }
        }


class InterpolationRequest(BaseModel):
    model_type: str = Field("skip", description="Type of model to use")
    start_vector: List[float] = Field(..., description="Starting latent vector")
    end_vector: List[float] = Field(..., description="Ending latent vector")
    num_steps: int = Field(5, description="Number of interpolation steps", ge=2, le=20)
    slice_axis: int = Field(0, description="Axis for 2D slice visualization", ge=0, le=2)
    slice_position: float = Field(
        0.5, description="Position for slice (0.0-1.0)", ge=0.0, le=1.0
    )
    
    @validator('start_vector', 'end_vector')
    def validate_vector_length(cls, v):
        if len(v) != 100:  # Default latent dimension
            raise ValueError('Vector must have length 100')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "model_type": "skip",
                "start_vector": [0.0] * 100,
                "end_vector": [0.1] * 100,
                "num_steps": 5,
                "slice_axis": 0,
                "slice_position": 0.5
            }
        }


class ModelInfo(BaseModel):
    model_type: str
    description: str
    parameters: Dict[str, Union[int, float, str]]
    version: Optional[str] = None
    source: Optional[str] = None


class GenerationResponse(BaseModel):
    request_id: str
    image_url: str
    model_type: str
    generation_time: float
    parameters: Dict[str, Union[int, float, str, List[float]]]
    model_data_url: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    class Config:
        schema_extra = {
            "example": {
                "request_id": "abcd1234",
                "image_url": "/outputs/output_abcd1234_20220101_120000.png",
                "model_type": "skip",
                "generation_time": 1.25,
                "parameters": {
                    "num_samples": 4,
                    "noise_dim": 100,
                    "seed": 42,
                    "slice_axis": 0,
                    "slice_position": 0.5
                },
                "model_data_url": "/outputs/model_abcd1234_20220101_120000.npy",
                "created_at": "2022-01-01T12:00:00.000000"
            }
        }


class BatchGenerationRequest(BaseModel):
    model_type: str = Field("skip", description="Type of model to use")
    num_batches: int = Field(5, description="Number of batches to generate", ge=1, le=20)
    samples_per_batch: int = Field(4, description="Samples per batch", ge=1, le=16)
    noise_dim: int = Field(100, description="Dimension of noise vector")
    seed_start: Optional[int] = Field(None, description="Starting seed (will be incremented)")
    slice_axis: int = Field(0, description="Axis for 2D slice visualization", ge=0, le=2)
    slice_position: float = Field(0.5, description="Position for slice (0.0-1.0)", ge=0.0, le=1.0)
    
    class Config:
        schema_extra = {
            "example": {
                "model_type": "skip",
                "num_batches": 5,
                "samples_per_batch": 4,
                "noise_dim": 100,
                "seed_start": 1000,
                "slice_axis": 0,
                "slice_position": 0.5
            }
        }


class BatchGenerationResponse(BaseModel):
    batch_id: str
    num_batches: int
    samples_per_batch: int
    status: str
    batches_completed: int
    total_samples: int
    created_at: str
    estimated_completion_time: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "batch_id": "batch_abcd1234",
                "num_batches": 5,
                "samples_per_batch": 4,
                "status": "in_progress",
                "batches_completed": 2,
                "total_samples": 20,
                "created_at": "2022-01-01T12:00:00.000000",
                "estimated_completion_time": "2022-01-01T12:05:00.000000"
            }
        }


class ModifyVectorRequest(BaseModel):
    base_vector: List[float] = Field(..., description="Base latent vector")
    dimension: int = Field(..., description="Dimension to modify", ge=0, lt=100)
    value: float = Field(..., description="New value for the dimension")
    model_type: str = Field("skip", description="Type of model to use")
    
    @validator('base_vector')
    def validate_vector_length(cls, v):
        if len(v) != 100:  # Default latent dimension
            raise ValueError('Vector must have length 100')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "base_vector": [0.0] * 100,
                "dimension": 5,
                "value": 2.0,
                "model_type": "skip"
            }
        }


# Helper functions
def get_model(model_type: str):
    """Get or load a model of the specified type."""
    if model_type in MODEL_CACHE:
        return MODEL_CACHE[model_type]
    
    if not MODULES_AVAILABLE:
        raise HTTPException(status_code=500, detail="DeepSculpt modules not available")
    
    try:
        # Try to load from MLflow first
        manager = Manager()
        model = manager.load_mlflow_model(stage="Production")
        
        # If no model found in MLflow, create a new one
        if model is None:
            model = ModelFactory.create_generator(model_type=model_type)
        
        MODEL_CACHE[model_type] = model
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


def cleanup_old_outputs(max_files=1000):
    """Delete oldest output files when max_files is reached."""
    try:
        # Get all files in output directory
        files = []
        for file_path in Path(OUTPUT_DIR).glob("*.*"):
            if file_path.is_file():
                files.append((file_path, file_path.stat().st_mtime))
        
        # If we have too many files, delete the oldest ones
        if len(files) > max_files:
            # Sort by modification time (oldest first)
            files.sort(key=lambda x: x[1])
            
            # Delete the oldest files to get back to the limit
            files_to_delete = files[:len(files) - max_files]
            for file_path, _ in files_to_delete:
                os.remove(file_path)
                logger.info(f"Deleted old output file: {file_path}")
    
    except Exception as e:
        logger.error(f"Error cleaning up old outputs: {e}")


def save_to_history(entry: Dict[str, Any]):
    """Save a generation to history."""
    try:
        # Create a unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"history_{entry['request_id']}_{timestamp}.json"
        file_path = os.path.join(HISTORY_DIR, filename)
        
        # Save the entry
        with open(file_path, "w") as f:
            json.dump(entry, f)
        
        # Cleanup old history entries if needed
        history_files = list(Path(HISTORY_DIR).glob("history_*.json"))
        if len(history_files) > MAX_HISTORY_ENTRIES:
            # Sort by modification time (oldest first)
            history_files.sort(key=lambda x: x.stat().st_mtime)
            
            # Delete the oldest files
            files_to_delete = history_files[:len(history_files) - MAX_HISTORY_ENTRIES]
            for file_path in files_to_delete:
                os.remove(file_path)
                logger.info(f"Deleted old history entry: {file_path}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error saving to history: {e}")
        return False


def generate_visualization(
    model, 
    num_samples: int = 1, 
    noise_dim: int = 100, 
    seed: Optional[int] = None, 
    noise_vectors: Optional[List[List[float]]] = None,
    slice_axis: int = 0, 
    slice_position: float = 0.5,
    visual_style: str = "default",
    save_3d: bool = False
):
    """Generate samples and create visualization."""
    # Set seed for reproducibility if provided
    if seed is not None:
        tf.random.set_seed(seed)
        np.random.seed(seed)

    # Generate or use provided noise
    if noise_vectors is not None:
        # Convert list of vectors to tensor
        noise = tf.convert_to_tensor(noise_vectors, dtype=tf.float32)
    else:
        # Generate random noise
        noise = tf.random.normal([num_samples, noise_dim])

    # Generate samples
    samples = model(noise, training=False)
    
    # Create output filename with unique ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    req_id = str(uuid.uuid4())[:8]
    
    # Create figure for visualization
    fig_size = min(15, 3 * (num_samples ** 0.5))
    fig = plt.figure(figsize=(fig_size, fig_size))
    
    # Determine grid size
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    
    # Determine if we're working with color or monochrome
    is_color = samples.shape[-1] > 3
    
    for i in range(num_samples):
        if i >= grid_size * grid_size:
            break
            
        plt.subplot(grid_size, grid_size, i + 1)
        
        # Get slice based on axis and position
        sample = samples[i].numpy()
        slice_idx = int(slice_position * sample.shape[slice_axis])
        
        if slice_axis == 0:
            slice_data = sample[slice_idx, :, :, :3]  # Use first 3 channels for RGB
        elif slice_axis == 1:
            slice_data = sample[:, slice_idx, :, :3]
        else:
            slice_data = sample[:, :, slice_idx, :3]
        
        # Apply different visual styles
        if visual_style == "default":
            # Normalize for display
            slice_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data) + 1e-8)
        elif visual_style == "heatmap":
            # Create a heatmap style
            plt.set_cmap('hot')
            slice_data = np.mean(slice_data, axis=2)  # Convert to grayscale
        elif visual_style == "contour":
            # Create a contour style
            plt.set_cmap('viridis')
            slice_data = np.mean(slice_data, axis=2)  # Convert to grayscale
            plt.contourf(slice_data, levels=10)
            plt.axis('off')
            continue  # Skip the imshow below
        
        plt.imshow(slice_data)
        plt.title(f"Sample {i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    
    # Save to file
    output_path = os.path.join(OUTPUT_DIR, f"output_{req_id}_{timestamp}.png")
    plt.savefig(output_path)
    plt.close(fig)
    
    # Save 3D model data if requested
    model_data_path = None
    if save_3d:
        model_data_path = os.path.join(OUTPUT_DIR, f"model_{req_id}_{timestamp}.npy")
        np.save(model_data_path, samples.numpy())
    
    # Convert noise to list for saving in history
    noise_list = noise.numpy().tolist()
    
    return {
        "output_path": output_path,
        "model_data_path": model_data_path,
        "request_id": req_id,
        "noise_vectors": noise_list,
        "timestamp": timestamp
    }


async def generate_batch(batch_id: str, model_type: str, num_batches: int, samples_per_batch: int,
                       noise_dim: int, seed_start: Optional[int], slice_axis: int, slice_position: float):
    """Background task to generate multiple batches."""
    # Get the model
    try:
        model = get_model(model_type)
        
        # Create a status file
        status_file = os.path.join(OUTPUT_DIR, f"{batch_id}_status.json")
        status = {
            "batch_id": batch_id,
            "status": "in_progress",
            "batches_completed": 0,
            "total_batches": num_batches,
            "samples_per_batch": samples_per_batch,
            "total_samples": num_batches * samples_per_batch,
            "created_at": datetime.now().isoformat(),
            "model_type": model_type,
            "results": []
        }
        
        # Save initial status
        with open(status_file, "w") as f:
            json.dump(status, f, indent=2)
        
        # Generate each batch
        start_time = time.time()
        current_seed = seed_start
        
        for batch in range(num_batches):
            try:
                # Generate this batch
                seed = None if current_seed is None else current_seed + batch
                
                result = generate_visualization(
                    model=model,
                    num_samples=samples_per_batch,
                    noise_dim=noise_dim,
                    seed=seed,
                    slice_axis=slice_axis,
                    slice_position=slice_position
                )
                
                # Add to results
                status["results"].append({
                    "batch_index": batch,
                    "request_id": result["request_id"],
                    "image_url": f"/outputs/{os.path.basename(result['output_path'])}",
                    "seed": seed
                })
                
                # Update status
                status["batches_completed"] = batch + 1
                
                # Estimate completion time
                elapsed_time = time.time() - start_time
                avg_time_per_batch = elapsed_time / (batch + 1)
                remaining_batches = num_batches - (batch + 1)
                estimated_remaining_time = avg_time_per_batch * remaining_batches
                
                if remaining_batches > 0:
                    estimated_completion_time = datetime.now().timestamp() + estimated_remaining_time
                    status["estimated_completion_time"] = datetime.fromtimestamp(estimated_completion_time).isoformat()
                
                # Save updated status
                with open(status_file, "w") as f:
                    json.dump(status, f, indent=2)
                
                # Give the system a short break
                time.sleep(0.1)
            
            except Exception as e:
                logger.error(f"Error generating batch {batch}: {e}")
                # Continue with next batch
        
        # Mark as completed
        status["status"] = "completed"
        status["completed_at"] = datetime.now().isoformat()
        
        # Save final status
        with open(status_file, "w") as f:
            json.dump(status, f, indent=2)
        
        logger.info(f"Batch generation completed: {batch_id}")
        
    except Exception as e:
        logger.error(f"Error in batch generation task: {e}")
        
        # Update status file with error if it exists
        try:
            if os.path.exists(status_file):
                with open(status_file, "r") as f:
                    status = json.load(f)
                
                status["status"] = "error"
                status["error"] = str(e)
                
                with open(status_file, "w") as f:
                    json.dump(status, f, indent=2)
        except:
            pass


# API endpoints
@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to docs."""
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "modules_available": MODULES_AVAILABLE
    }
    
    # Check if we can load a model
    if MODULES_AVAILABLE:
        try:
            model = get_model("skip")
            status["model_loaded"] = True
        except:
            status["model_loaded"] = False
    else:
        status["model_loaded"] = False
    
    return status


@app.get("/models", response_model=List[ModelInfo])
async def list_models(api_key: str = Depends(get_api_key)):
    """List all available models."""
    # Basic model definitions
    models = [
        {
            "model_type": "simple",
            "description": "Simple 3D generator model",
            "parameters": {
                "void_dim": 64,
                "noise_dim": 100,
                "color_mode": 1
            }
        },
        {
            "model_type": "complex",
            "description": "Complex 3D generator with enhanced features",
            "parameters": {
                "void_dim": 64,
                "noise_dim": 100,
                "color_mode": 1
            }
        },
        {
            "model_type": "skip",
            "description": "3D generator with skip connections",
            "parameters": {
                "void_dim": 64,
                "noise_dim": 100,
                "color_mode": 1
            }
        },
        {
            "model_type": "monochrome",
            "description": "Monochrome 3D generator",
            "parameters": {
                "void_dim": 64,
                "noise_dim": 100,
                "color_mode": 0
            }
        },
        {
            "model_type": "autoencoder",
            "description": "Adversarial autoencoder for 3D generation",
            "parameters": {
                "latent_dim": 100,
                "void_dim": 64,
                "color_mode": 1
            }
        },
        {
            "model_type": "residual",
            "description": "3D generator with residual blocks",
            "parameters": {
                "void_dim": 64,
                "noise_dim": 100,
                "color_mode": 1,
                "use_attention": True
            }
        }
    ]
    
    # Try to add MLflow models if available
    if MODULES_AVAILABLE:
        try:
            # Check if MLflow is available
            import mlflow
            from mlflow.tracking import MlflowClient
            
            # Set tracking URI
            mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
            if mlflow_tracking_uri:
                mlflow.set_tracking_uri(mlflow_tracking_uri)
                
                # Get client
                client = MlflowClient()
                
                # Get model name
                model_name = os.environ.get("MLFLOW_MODEL_NAME")
                if model_name:
                    # Get model versions
                    versions = client.get_latest_versions(model_name)
                    
                    # Add MLflow models to the list
                    for version in versions:
                        # Check if model already exists in the list
                        model_exists = False
                        for model in models:
                            if model["model_type"] == version.name:
                                # Update existing model info
                                model["version"] = version.version
                                model["source"] = "mlflow"
                                model_exists = True
                                break
                        
                        if not model_exists:
                            # Add new model
                            mlflow_model = {
                                "model_type": version.name,
                                "description": f"MLflow model {version.name} (version {version.version})",
                                "parameters": {
                                    "version": version.version,
                                    "stage": version.current_stage
                                },
                                "version": version.version,
                                "source": "mlflow"
                            }
                            models.append(mlflow_model)
        except:
            # MLflow not available or error occurred
            pass
    
    return models


@app.get("/model/{model_type}", response_model=ModelInfo)
async def get_model_info(model_type: str, api_key: str = Depends(get_api_key)):
    """Get information about a specific model."""
    for model in await list_models(api_key):
        if model["model_type"] == model_type:
            return model
    
    raise HTTPException(status_code=404, detail=f"Model type '{model_type}' not found")


@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest, api_key: str = Depends(get_api_key)):
    """Generate 3D samples and return visualization."""
    try:
        logger.info(f"Generation request: {request.model_type}, {request.num_samples} samples")
        
        # Get the model
        model = get_model(request.model_type)
        
        # Measure generation time
        start_time = time.time()
        
        # Generate visualization
        result = generate_visualization(
            model=model,
            num_samples=request.num_samples,
            noise_dim=request.noise_dim,
            seed=request.seed,
            slice_axis=request.slice_axis,
            slice_position=request.slice_position,
            visual_style=request.visual_style,
            save_3d=request.save_3d
        )
        
        # Calculate generation time
        generation_time = time.time() - start_time
        
        # Get relative URL
        image_url = f"/outputs/{os.path.basename(result['output_path'])}"
        model_data_url = None
        
        if result['model_data_path']:
            model_data_url = f"/outputs/{os.path.basename(result['model_data_path'])}"
        
        # Create response
        response = {
            "request_id": result["request_id"],
            "image_url": image_url,
            "model_type": request.model_type,
            "generation_time": generation_time,
            "parameters": {
                "num_samples": request.num_samples,
                "noise_dim": request.noise_dim,
                "seed": request.seed,
                "slice_axis": request.slice_axis,
                "slice_position": request.slice_position,
                "noise_vectors": result["noise_vectors"],
                "visual_style": request.visual_style,
                "batch_id": request.batch_id
            },
            "model_data_url": model_data_url,
            "created_at": datetime.now().isoformat()
        }
        
        # Save to history
        save_to_history(response)
        
        # Clean up old outputs if needed
        cleanup_old_outputs()
        
        return response
    
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/interpolate", response_model=GenerationResponse)
async def interpolate(request: InterpolationRequest, api_key: str = Depends(get_api_key)):
    """Interpolate between two latent vectors and generate visualizations."""
    try:
        logger.info(f"Interpolation request: {request.model_type}, {request.num_steps} steps")
        
        # Get the model
        model = get_model(request.model_type)
        
        # Measure generation time
        start_time = time.time()
        
        # Generate interpolated vectors
        start_vector = np.array(request.start_vector, dtype=np.float32)
        end_vector = np.array(request.end_vector, dtype=np.float32)
        
        interpolated_vectors = []
        for t in np.linspace(0, 1, request.num_steps):
            interpolated_vector = start_vector * (1 - t) + end_vector * t
            interpolated_vectors.append(interpolated_vector)
        
        # Generate visualization
        result = generate_visualization(
            model=model,
            num_samples=request.num_steps,
            noise_vectors=interpolated_vectors,
            slice_axis=request.slice_axis,
            slice_position=request.slice_position,
            save_3d=True  # Always save for interpolations
        )
        
        # Calculate generation time
        generation_time = time.time() - start_time
        
        # Get relative URL
        image_url = f"/outputs/{os.path.basename(result['output_path'])}"
        model_data_url = f"/outputs/{os.path.basename(result['model_data_path'])}"
        
        # Create response
        response = {
            "request_id": result["request_id"],
            "image_url": image_url,
            "model_type": request.model_type,
            "generation_time": generation_time,
            "parameters": {
                "num_steps": request.num_steps,
                "start_vector": request.start_vector,
                "end_vector": request.end_vector,
                "slice_axis": request.slice_axis,
                "slice_position": request.slice_position,
                "interpolation": True,
                "noise_vectors": result["noise_vectors"]
            },
            "model_data_url": model_data_url,
            "created_at": datetime.now().isoformat()
        }
        
        # Save to history
        save_to_history(response)
        
        # Clean up old outputs if needed
        cleanup_old_outputs()
        
        return response
    
    except Exception as e:
        logger.error(f"Interpolation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Interpolation failed: {str(e)}")


@app.post("/modify-vector", response_model=GenerationResponse)
async def modify_vector(request: ModifyVectorRequest, api_key: str = Depends(get_api_key)):
    """Modify a specific dimension of a latent vector and generate the result."""
    try:
        logger.info(f"Vector modification request: dimension {request.dimension} to {request.value}")
        
        # Get the model
        model = get_model(request.model_type)
        
        # Measure generation time
        start_time = time.time()
        
        # Create modified vector
        modified_vector = np.array(request.base_vector, dtype=np.float32)
        modified_vector[request.dimension] = request.value
        
        # Generate visualization
        result = generate_visualization(
            model=model,
            num_samples=1,
            noise_vectors=[modified_vector],
            save_3d=True  # Always save for vector modifications
        )
        
        # Calculate generation time
        generation_time = time.time() - start_time
        
        # Get relative URL
        image_url = f"/outputs/{os.path.basename(result['output_path'])}"
        model_data_url = f"/outputs/{os.path.basename(result['model_data_path'])}"
        
        # Create response
        response = {
            "request_id": result["request_id"],
            "image_url": image_url,
            "model_type": request.model_type,
            "generation_time": generation_time,
            "parameters": {
                "base_vector": request.base_vector,
                "dimension": request.dimension,
                "value": request.value,
                "modified_vector": modified_vector.tolist(),
                "noise_vectors": result["noise_vectors"]
            },
            "model_data_url": model_data_url,
            "created_at": datetime.now().isoformat()
        }
        
        # Save to history
        save_to_history(response)
        
        # Clean up old outputs if needed
        cleanup_old_outputs()
        
        return response
    
    except Exception as e:
        logger.error(f"Vector modification failed: {e}")
        raise HTTPException(status_code=500, detail=f"Vector modification failed: {str(e)}")


@app.post("/batch", response_model=BatchGenerationResponse)
async def generate_batch_request(
    request: BatchGenerationRequest, 
    background_tasks: BackgroundTasks,
    api_key: str = Depends(get_api_key)
):
    """
    Start a batch generation job.
    
    This creates multiple batches of samples in the background and returns a batch ID
    that can be used to check the status of the job.
    """
    try:
        # Create a batch ID
        batch_id = f"batch_{str(uuid.uuid4())[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Start batch generation in the background
        background_tasks.add_task(
            generate_batch,
            batch_id=batch_id,
            model_type=request.model_type,
            num_batches=request.num_batches,
            samples_per_batch=request.samples_per_batch,
            noise_dim=request.noise_dim,
            seed_start=request.seed_start,
            slice_axis=request.slice_axis,
            slice_position=request.slice_position
        )
        
        # Return immediate response with batch ID
        return {
            "batch_id": batch_id,
            "num_batches": request.num_batches,
            "samples_per_batch": request.samples_per_batch,
            "status": "started",
            "batches_completed": 0,
            "total_samples": request.num_batches * request.samples_per_batch,
            "created_at": datetime.now().isoformat(),
            "estimated_completion_time": None
        }
    
    except Exception as e:
        logger.error(f"Batch request failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch request failed: {str(e)}")


@app.get("/batch/{batch_id}", response_model=Dict[str, Any])
async def get_batch_status(batch_id: str, api_key: str = Depends(get_api_key)):
    """Get the status of a batch generation job."""
    try:
        # Check if status file exists
        status_file = os.path.join(OUTPUT_DIR, f"{batch_id}_status.json")
        
        if not os.path.exists(status_file):
            raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found")
        
        # Read status file
        with open(status_file, "r") as f:
            status = json.load(f)
        
        return status
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting batch status: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting batch status: {str(e)}")


@app.get("/image/{request_id}")
async def get_image(request_id: str, api_key: str = Depends(get_api_key)):
    """Get a generated image by request ID."""
    # Find the image file
    for file in os.listdir(OUTPUT_DIR):
        if request_id in file and (file.endswith(".png") or file.endswith(".jpg")):
            return FileResponse(os.path.join(OUTPUT_DIR, file))
    
    raise HTTPException(status_code=404, detail=f"Image for request ID '{request_id}' not found")


@app.get("/model/{request_id}")
async def get_model_data(request_id: str, api_key: str = Depends(get_api_key)):
    """Get the 3D model data for a request."""
    # Find the model data file
    for file in os.listdir(OUTPUT_DIR):
        if request_id in file and file.endswith(".npy"):
            return FileResponse(os.path.join(OUTPUT_DIR, file))
    
    raise HTTPException(status_code=404, detail=f"Model data for request ID '{request_id}' not found")


@app.get("/history", response_model=List[Dict[str, Any]])
async def get_history(limit: int = 10, skip: int = 0, api_key: str = Depends(get_api_key)):
    """Get generation history."""
    try:
        # Get all history files
        history_files = list(Path(HISTORY_DIR).glob("history_*.json"))
        
        # Sort by modification time (newest first)
        history_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Apply pagination
        paginated_files = history_files[skip:skip + limit]
        
        # Load history entries
        history = []
        for file_path in paginated_files:
            try:
                with open(file_path, "r") as f:
                    entry = json.load(f)
                    history.append(entry)
            except Exception as e:
                logger.error(f"Error loading history entry {file_path}: {e}")
        
        return history
    
    except Exception as e:
        logger.error(f"Error getting history: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting history: {str(e)}")


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    api_key: str = Depends(get_api_key)
):
    """Upload a file for processing."""
    try:
        # Create unique filename
        ext = os.path.splitext(file.filename)[1]
        unique_filename = f"{str(uuid.uuid4())[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        
        # Save the file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return {
            "filename": unique_filename,
            "original_filename": file.filename,
            "file_path": file_path,
            "upload_time": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)