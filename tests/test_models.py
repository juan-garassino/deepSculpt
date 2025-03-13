"""
Tests for the DeepSculpt model architectures.
"""

import pytest
import tensorflow as tf
import numpy as np
import os
import sys

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import ModelFactory, add_regularization


@pytest.fixture(scope="module")
def setup_env():
    """Setup environment variables for tests."""
    os.environ["VOID_DIM"] = "32"  # Smaller for faster tests
    os.environ["NOISE_DIM"] = "50"  # Smaller for faster tests
    os.environ["COLOR"] = "1"
    
    # Define test parameters
    params = {
        "void_dim": 32,
        "noise_dim": 50,
        "color_mode": 1,
    }
    return params


def test_model_factory_exists():
    """Test that ModelFactory class exists."""
    assert hasattr(ModelFactory, "create_generator")
    assert hasattr(ModelFactory, "create_discriminator")


@pytest.mark.parametrize("model_type", ["simple", "complex", "skip", "monochrome", "autoencoder"])
def test_create_generator(setup_env, model_type):
    """Test creating generators of different types."""
    params = setup_env
    
    # Create generator
    generator = ModelFactory.create_generator(
        model_type=model_type,
        void_dim=params["void_dim"],
        noise_dim=params["noise_dim"],
        color_mode=params["color_mode"]
    )
    
    # Check that it's a Keras model
    assert isinstance(generator, tf.keras.Model)
    
    # Test forward pass with random noise
    batch_size = 2
    noise = tf.random.normal([batch_size, params["noise_dim"]])
    output = generator(noise, training=False)
    
    # Check output shape
    if model_type == "autoencoder":
        # Autoencoder has a different output shape
        expected_shape = (batch_size, 32, 32, 32, 32)  # Example shape, adjust as needed
    else:
        output_channels = 6 if params["color_mode"] == 1 else 3
        expected_shape = (batch_size, params["void_dim"], params["void_dim"], params["void_dim"], output_channels)
    
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"


@pytest.mark.parametrize("model_type", ["simple", "complex", "skip", "monochrome"])
def test_create_discriminator(setup_env, model_type):
    """Test creating discriminators of different types."""
    params = setup_env
    
    # Create discriminator
    discriminator = ModelFactory.create_discriminator(
        model_type=model_type,
        void_dim=params["void_dim"],
        noise_dim=params["noise_dim"],
        color_mode=params["color_mode"]
    )
    
    # Check that it's a Keras model
    assert isinstance(discriminator, tf.keras.Model)
    
    # Test forward pass with random input
    batch_size = 2
    output_channels = 6 if params["color_mode"] == 1 else 3
    input_shape = (batch_size, params["void_dim"], params["void_dim"], params["void_dim"], output_channels)
    inputs = tf.random.normal(input_shape)
    output = discriminator(inputs, training=False)
    
    # Check output shape
    assert output.shape == (batch_size, 1), f"Expected shape {(batch_size, 1)}, got {output.shape}"


def test_add_regularization(setup_env):
    """Test adding dropout regularization to a model."""
    params = setup_env
    
    # Create a model
    generator = ModelFactory.create_generator(
        model_type="simple",
        void_dim=params["void_dim"],
        noise_dim=params["noise_dim"]
    )
    
    # Count the number of dropout layers
    original_dropout_count = len([layer for layer in generator.layers if isinstance(layer, tf.keras.layers.Dropout)])
    
    # Add regularization
    regularized_generator = add_regularization(generator, dropout_rate=0.3)
    
    # Count the number of dropout layers again
    new_dropout_count = len([layer for layer in regularized_generator.layers if isinstance(layer, tf.keras.layers.Dropout)])
    
    # Check that dropout layers were added
    assert new_dropout_count > original_dropout_count


def test_model_compilation():
    """Test that models can be compiled."""
    # Create models
    generator = ModelFactory.create_generator(model_type="simple")
    discriminator = ModelFactory.create_discriminator(model_type="simple")
    
    # Compile models
    generator.compile(optimizer='adam')
    discriminator.compile(optimizer='adam', loss='binary_crossentropy')
    
    # Check that they compiled successfully
    assert generator.optimizer is not None
    assert discriminator.optimizer is not None
    assert discriminator.loss is not None


def test_model_save_and_load(tmp_path, setup_env):
    """Test saving and loading models."""
    params = setup_env
    
    # Create a model
    generator = ModelFactory.create_generator(
        model_type="simple",
        void_dim=params["void_dim"],
        noise_dim=params["noise_dim"]
    )
    
    # Save the model
    save_path = str(tmp_path / "test_model")
    generator.save(save_path)
    
    # Check that the model was saved
    assert os.path.exists(save_path)
    
    # Load the model
    loaded_model = tf.keras.models.load_model(save_path)
    
    # Check that the loaded model has the same architecture
    assert len(generator.layers) == len(loaded_model.layers)
    
    # Test forward pass with the same input
    noise = tf.random.normal([1, params["noise_dim"]])
    original_output = generator(noise, training=False)
    loaded_output = loaded_model(noise, training=False)
    
    # Check that the outputs are the same
    np.testing.assert_allclose(original_output, loaded_output, rtol=1e-5, atol=1e-5)