"""
FastAPI server for DeepSculpt model inference.

This module provides API endpoints to:
1. Generate 3D models and visualizations
2. Get information about available models
3. Manage model parameters and settings

To run the server:
    uvicorn api:app --reload
"""

import os
import io
import base64
import uuid
import json
import numpy as np
import tensorflow as tf
from typing import List, Dict, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field
import matplotlib.pyplot as plt

from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile, Form, Query
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Import DeepSculpt modules
from models import ModelFactory
from workflow import Manager

# Create FastAPI app
app = FastAPI(
    title="DeepSculpt API",
    description="API for generating 3D models using DeepSculpt",
    version="1.0.0"
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
MODEL_CACHE = {}  # Cache for loaded models
UPLOAD_DIR = "./uploads"
OUTPUT_DIR = "./outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mount static files directory for serving images
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


# Pydantic models for request/response validation
class GenerationRequest(BaseModel):
    model_type: str = Field("skip", description="Type of model to use")
    noise_dim: int = Field(100, description="Dimension of noise vector")
    num_samples: int = Field(1, description="Number of samples to generate", ge=1, le=16)
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    slice_axis: int = Field(0, description="Axis for 2D slice visualization", ge=0, le=2)
    slice_position: Optional[float] = Field(
        0.5, description="Position for slice (0.0-1.0)", ge=0.0, le=1.0
    )

class ModelInfo(BaseModel):
    model_type: str
    description: str
    parameters: Dict[str, Union[int, float, str]]

class GenerationResponse(BaseModel):
    request_id: str
    image_url: str
    model_type: str
    generation_time: float
    parameters: Dict[str, Union[int, float, str]]


# Helper functions
def get_model(model_type: str):
    """Get or load a model of the specified type."""
    if model_type in MODEL_CACHE:
        return MODEL_CACHE[model_type]
    
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
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

def generate_visualization(
    model, num_samples: int = 1, noise_dim: int = 100, 
    seed: Optional[int] = None, slice_axis: int = 0, 
    slice_position: float = 0.5
):
    """Generate samples and create visualization."""
    # Set seed for reproducibility if provided
    if seed is not None:
        tf.random.set_seed(seed)
        np.random.seed(seed)
    
    # Generate noise
    noise = tf.random.normal([num_samples, noise_dim])
    
    # Generate samples
    samples = model(noise, training=False)
    
    # Create figure for visualization
    fig = plt.figure(figsize=(12, 12))
    
    # Determine grid size
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    
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
        
        # Normalize for display
        slice_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data) + 1e-8)
        
        plt.imshow(slice_data)
        plt.title(f"Sample {i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    
    # Save to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    req_id = str(uuid.uuid4())[:8]
    output_path = os.path.join(OUTPUT_DIR, f"output_{req_id}_{timestamp}.png")
    plt.savefig(output_path)
    plt.close(fig)
    
    return output_path, req_id


# API endpoints
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "DeepSculpt API is running",
        "version": "1.0.0",
        "endpoints": {
            "/generate": "Generate 3D samples",
            "/models": "List available models",
            "/model/{model_type}": "Get model information"
        }
    }

@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List all available models."""
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
        }
    ]
    return models

@app.get("/model/{model_type}", response_model=ModelInfo)
async def get_model_info(model_type: str):
    """Get information about a specific model."""
    for model in await list_models():
        if model["model_type"] == model_type:
            return model
    
    raise HTTPException(status_code=404, detail=f"Model type '{model_type}' not found")

@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    """Generate 3D samples and return visualization."""
    try:
        # Get the model
        model = get_model(request.model_type)
        
        # Measure generation time
        start_time = datetime.now()
        
        # Generate visualization
        output_path, req_id = generate_visualization(
            model=model,
            num_samples=request.num_samples,
            noise_dim=request.noise_dim,
            seed=request.seed,
            slice_axis=request.slice_axis,
            slice_position=request.slice_position
        )
        
        # Calculate generation time
        generation_time = (datetime.now() - start_time).total_seconds()
        
        # Get relative URL
        image_url = f"/outputs/{os.path.basename(output_path)}"
        
        # Return response
        return {
            "request_id": req_id,
            "image_url": image_url,
            "model_type": request.model_type,
            "generation_time": generation_time,
            "parameters": {
                "num_samples": request.num_samples,
                "noise_dim": request.noise_dim,
                "seed": request.seed,
                "slice_axis": request.slice_axis,
                "slice_position": request.slice_position
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.get("/image/{request_id}")
async def get_image(request_id: str):
    """Get a generated image by request ID."""
    # Find the image file
    for file in os.listdir(OUTPUT_DIR):
        if request_id in file and (file.endswith(".png") or file.endswith(".jpg")):
            return FileResponse(os.path.join(OUTPUT_DIR, file))
    
    raise HTTPException(status_code=404, detail=f"Image for request ID '{request_id}' not found")


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)