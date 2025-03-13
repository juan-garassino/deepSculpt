"""
Pytest configuration file for DeepSculpt tests.
"""

import pytest
import os
import sys
import tensorflow as tf
import numpy as np
import tempfile
import shutil

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session", autouse=True)
def disable_gpu():
    """Disable GPU for faster tests."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    # Verify that GPU is disabled
    devices = tf.config.list_physical_devices('GPU')
    assert len(devices) == 0, "GPU should be disabled for tests"


@pytest.fixture(scope="session", autouse=True)
def set_tf_random_seed():
    """Set TensorFlow random seed for reproducibility."""
    tf.random.set_seed(42)
    np.random.seed(42)


@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory with test data."""
    # Create temporary directory
    test_dir = tempfile.mkdtemp(prefix="deepsculpt_test_")
    
    # Create test data
    volume_data = np.random.rand(5, 16, 16, 16).astype(np.float32)
    material_data = np.random.randint(0, 6, (5, 16, 16, 16)).astype(np.int32)
    
    # Save test data
    vol_path = os.path.join(test_dir, "volume_data[2023-01-01]chunk[1].npy")
    mat_path = os.path.join(test_dir, "material_data[2023-01-01]chunk[1].npy")
    
    np.save(vol_path, volume_data)
    np.save(mat_path, material_data)
    
    # Create subdirectories
    os.makedirs(os.path.join(test_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "snapshots"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "outputs"), exist_ok=True)
    
    yield test_dir
    
    # Cleanup
    shutil.rmtree(test_dir)


@pytest.fixture(scope="session")
def small_test_model():
    """Create a small test model for faster tests."""
    # Create a very small generator model for testing
    generator = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(64),
        tf.keras.layers.Reshape((4, 4, 4, 1))
    ])
    
    return generator


@pytest.fixture(scope="session")
def setup_global_env():
    """Setup global environment variables for all tests."""
    # Save original environment
    original_env = dict(os.environ)
    
    # Set test environment variables
    os.environ["VOID_DIM"] = "16"  # Smaller for faster tests
    os.environ["NOISE_DIM"] = "10"  # Smaller for faster tests
    os.environ["COLOR"] = "1"
    os.environ["INSTANCE"] = "0"
    os.environ["MINIBATCH_SIZE"] = "2"
    os.environ["EPOCHS"] = "2"
    os.environ["MODEL_CHECKPOINT"] = "1"
    os.environ["PICTURE_SNAPSHOT"] = "1"
    os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
    os.environ["MLFLOW_EXPERIMENT"] = "test_experiment"
    os.environ["MLFLOW_MODEL_NAME"] = "test_model"
    os.environ["PREFECT_FLOW_NAME"] = "test_workflow"
    os.environ["PREFECT_BACKEND"] = "development"
    os.environ["DEEPSCULPT_API_URL"] = "http://localhost:8000"
    os.environ["TELEGRAM_BOT_TOKEN"] = "test_token"
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)