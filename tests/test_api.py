"""
Tests for the DeepSculpt FastAPI server.
"""

import pytest
import os
import sys
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the API app
import api
from api import app


@pytest.fixture
def client():
    """Create a TestClient for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    import tensorflow as tf
    import numpy as np
    
    # Create a simple mock model that returns fixed data
    def mock_call(inputs, training=None):
        batch_size = inputs.shape[0]
        return tf.ones((batch_size, 64, 64, 64, 3))
    
    mock = MagicMock()
    mock.side_effect = mock_call
    
    return mock


def test_root_endpoint(client):
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "DeepSculpt API is running" in response.json()["message"]


def test_models_endpoint(client):
    """Test the /models endpoint."""
    response = client.get("/models")
    assert response.status_code == 200
    
    # Check response structure
    models = response.json()
    assert isinstance(models, list)
    assert len(models) > 0
    
    # Check that each model has the required fields
    for model in models:
        assert "model_type" in model
        assert "description" in model
        assert "parameters" in model


def test_model_info_endpoint(client):
    """Test the /model/{model_type} endpoint."""
    # Test with a valid model type
    response = client.get("/model/skip")
    assert response.status_code == 200
    assert response.json()["model_type"] == "skip"
    
    # Test with an invalid model type
    response = client.get("/model/nonexistent")
    assert response.status_code == 404


@patch("api.get_model")
@patch("api.generate_visualization")
def test_generate_endpoint(mock_generate_viz, mock_get_model, client, mock_model):
    """Test the /generate endpoint."""
    # Setup mocks
    mock_get_model.return_value = mock_model
    mock_generate_viz.return_value = ("/tmp/test.png", "test123")
    
    # Test with minimal parameters
    request_data = {
        "model_type": "skip",
        "num_samples": 1
    }
    
    response = client.post("/generate", json=request_data)
    assert response.status_code == 200
    
    # Check response structure
    result = response.json()
    assert "request_id" in result
    assert "image_url" in result
    assert "model_type" in result
    assert "generation_time" in result
    assert "parameters" in result
    
    # Test with all parameters
    request_data = {
        "model_type": "skip",
        "noise_dim": 100,
        "num_samples": 4,
        "seed": 42,
        "slice_axis": 0,
        "slice_position": 0.5
    }
    
    response = client.post("/generate", json=request_data)
    assert response.status_code == 200


@patch("fastapi.responses.FileResponse")
def test_get_image_endpoint(mock_file_response, client, tmp_path):
    """Test the /image/{request_id} endpoint."""
    # Create a test image file
    request_id = "test123"
    test_file = tmp_path / f"output_{request_id}_20230101.png"
    test_file.write_text("test image content")
    
    # Mock the OUTPUT_DIR in api.py
    original_output_dir = api.OUTPUT_DIR
    api.OUTPUT_DIR = str(tmp_path)
    
    try:
        # Setup mock
        mock_file_response.return_value = "file_response"
        
        # Test with valid request ID
        response = client.get(f"/image/{request_id}")
        
        # Check that FileResponse was called
        mock_file_response.assert_called()
        
        # Test with invalid request ID
        response = client.get("/image/nonexistent")
        assert response.status_code == 404
    
    finally:
        # Restore original OUTPUT_DIR
        api.OUTPUT_DIR = original_output_dir


def test_health_check_endpoint(client):
    """Test the /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "healthy"


@patch("api.tf.random.normal")
@patch("api.plt.savefig")
@patch("matplotlib.pyplot.subplot")
@patch("matplotlib.pyplot.imshow")
def test_generate_visualization(mock_imshow, mock_subplot, mock_savefig, mock_random_normal, mock_model):
    """Test the generate_visualization function."""
    # Import the function
    from api import generate_visualization
    
    # Setup mocks
    mock_random_normal.return_value = "noise_tensor"
    mock_savefig.return_value = None
    
    # Call the function
    output_path, req_id = generate_visualization(
        model=mock_model,
        num_samples=2,
        noise_dim=100,
        seed=42,
        slice_axis=0,
        slice_position=0.5
    )
    
    # Check that the output path and request ID were returned
    assert isinstance(output_path, str)
    assert output_path.endswith(".png")
    assert isinstance(req_id, str)
    
    # Check that the mocks were called
    assert mock_savefig.called


@patch("api.ModelFactory.create_generator")
@patch("api.Manager.load_mlflow_model")
def test_get_model(mock_load_mlflow, mock_create_generator):
    """Test the get_model function."""
    # Import the function
    from api import get_model
    
    # Setup mocks
    mock_model = MagicMock()
    mock_load_mlflow.return_value = mock_model
    mock_create_generator.return_value = mock_model
    
    # Test with MLflow model available
    model = get_model("skip")
    assert model == mock_model
    
    # Test with MLflow model not available
    mock_load_mlflow.return_value = None
    model = get_model("skip")
    assert model == mock_model
    mock_create_generator.assert_called_with(model_type="skip")