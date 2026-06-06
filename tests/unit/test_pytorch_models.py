"""
Comprehensive tests for PyTorch model architectures.
Tests equivalence with TensorFlow versions and validates functionality.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Any
import sys
import os

# Add the deepSculpt module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from deepsculpt.core.models.pytorch_models import (
    PyTorchModelFactory, SimpleGenerator, ComplexGenerator, SkipGenerator,
    MonochromeGenerator, AutoencoderGenerator, ProgressiveGenerator,
    ConditionalGenerator, SimpleDiscriminator, ComplexDiscriminator,
    SkipDiscriminator, MonochromeDiscriminator, AutoencoderDiscriminator,
    SpectralNormDiscriminator, ProgressiveDiscriminator, MultiScaleDiscriminator,
    ConditionalDiscriminator, SparseConv3d, SparseConvTranspose3d, SparseBatchNorm3d,
    ModelUtils, ModelArchitectureSearch, create_generator, create_discriminator
)

# Try to import TensorFlow models for comparison
try:
    from deepSculpt.models import ModelFactory as TFModelFactory
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class TestPyTorchGenerators:
    """Test PyTorch generator models."""
    
    @pytest.fixture
    def default_params(self):
        """Default parameters for testing."""
        return {
            "void_dim": 64,
            "noise_dim": 100,
            "color_mode": 1,
            "sparse": False
        }
    
    @pytest.fixture
    def test_input(self, default_params):
        """Test input tensor."""
        batch_size = 2
        return torch.randn(batch_size, default_params["noise_dim"])
    
    def test_simple_generator_creation(self, default_params):
        """Test SimpleGenerator creation and basic functionality."""
        generator = SimpleGenerator(**default_params)
        assert isinstance(generator, nn.Module)
        assert generator.void_dim == default_params["void_dim"]
        assert generator.noise_dim == default_params["noise_dim"]
        assert generator.color_mode == default_params["color_mode"]
    
    def test_simple_generator_forward(self, default_params, test_input):
        """Test SimpleGenerator forward pass."""
        generator = SimpleGenerator(**default_params)
        output = generator(test_input)
        
        expected_shape = (
            test_input.size(0),
            default_params["void_dim"],
            default_params["void_dim"],
            default_params["void_dim"],
            6 if default_params["color_mode"] == 1 else 3
        )
        
        assert output.shape == expected_shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_complex_generator_creation(self, default_params):
        """Test ComplexGenerator creation and functionality."""
        generator = ComplexGenerator(**default_params)
        assert isinstance(generator, nn.Module)
        
        # Test forward pass
        test_input = torch.randn(2, default_params["noise_dim"])
        output = generator(test_input)
        
        expected_shape = (
            2,
            default_params["void_dim"],
            default_params["void_dim"],
            default_params["void_dim"],
            6 if default_params["color_mode"] == 1 else 3
        )
        
        assert output.shape == expected_shape
    
    def test_skip_generator_creation(self, default_params):
        """Test SkipGenerator creation and functionality."""
        generator = SkipGenerator(**default_params)
        assert isinstance(generator, nn.Module)
        
        # Test forward pass
        test_input = torch.randn(2, default_params["noise_dim"])
        output = generator(test_input)
        
        expected_shape = (
            2,
            default_params["void_dim"],
            default_params["void_dim"],
            default_params["void_dim"],
            6 if default_params["color_mode"] == 1 else 3
        )
        
        assert output.shape == expected_shape
    
    def test_monochrome_generator_creation(self, default_params):
        """Test MonochromeGenerator creation and functionality."""
        params = default_params.copy()
        params["color_mode"] = 0
        
        generator = MonochromeGenerator(**params)
        assert isinstance(generator, nn.Module)
        assert generator.output_channels == 3
        
        # Test forward pass
        test_input = torch.randn(2, params["noise_dim"])
        output = generator(test_input)
        
        expected_shape = (
            2,
            params["void_dim"],
            params["void_dim"],
            params["void_dim"],
            3
        )
        
        assert output.shape == expected_shape
    
    def test_autoencoder_generator_creation(self, default_params):
        """Test AutoencoderGenerator creation and functionality."""
        generator = AutoencoderGenerator(**default_params)
        assert isinstance(generator, nn.Module)
        
        # Test forward pass
        test_input = torch.randn(2, default_params["noise_dim"])
        output = generator(test_input)
        
        # Autoencoder has different output size
        assert len(output.shape) == 5  # batch, depth, height, width, channels
        assert not torch.isnan(output).any()
    
    def test_progressive_generator_creation(self, default_params):
        """Test ProgressiveGenerator creation and functionality."""
        generator = ProgressiveGenerator(**default_params, max_resolution=128)
        assert isinstance(generator, nn.Module)
        assert generator.max_resolution == 128
        
        # Test forward pass
        test_input = torch.randn(2, default_params["noise_dim"])
        output = generator(test_input)
        
        assert len(output.shape) == 5
        assert not torch.isnan(output).any()
        
        # Test growing functionality
        initial_level = generator.current_level
        generator.grow()
        assert generator.current_level == initial_level + 1
    
    def test_conditional_generator_creation(self, default_params):
        """Test ConditionalGenerator creation and functionality."""
        condition_dim = 10
        generator = ConditionalGenerator(**default_params, condition_dim=condition_dim)
        assert isinstance(generator, nn.Module)
        
        # Test forward pass
        noise = torch.randn(2, default_params["noise_dim"])
        condition = torch.randint(0, condition_dim, (2,))
        output = generator(noise, condition)
        
        expected_shape = (
            2,
            default_params["void_dim"],
            default_params["void_dim"],
            default_params["void_dim"],
            6 if default_params["color_mode"] == 1 else 3
        )
        
        assert output.shape == expected_shape
    
    def test_sparse_generator_creation(self, default_params):
        """Test generator creation with sparse convolutions."""
        params = default_params.copy()
        params["sparse"] = True
        
        generator = SimpleGenerator(**params)
        assert isinstance(generator, nn.Module)
        
        # Check that sparse layers are used
        has_sparse_layers = any(isinstance(module, (SparseConv3d, SparseConvTranspose3d, SparseBatchNorm3d))
                               for module in generator.modules())
        assert has_sparse_layers
    
    @pytest.mark.parametrize("void_dim", [32, 64, 128])
    @pytest.mark.parametrize("color_mode", [0, 1])
    def test_generator_different_dimensions(self, void_dim, color_mode):
        """Test generators with different dimensions and color modes."""
        params = {
            "void_dim": void_dim,
            "noise_dim": 100,
            "color_mode": color_mode,
            "sparse": False
        }
        
        generator = SimpleGenerator(**params)
        test_input = torch.randn(1, params["noise_dim"])
        output = generator(test_input)
        
        expected_channels = 6 if color_mode == 1 else 3
        expected_shape = (1, void_dim, void_dim, void_dim, expected_channels)
        
        assert output.shape == expected_shape


class TestPyTorchDiscriminators:
    """Test PyTorch discriminator models."""
    
    @pytest.fixture
    def default_params(self):
        """Default parameters for testing."""
        return {
            "void_dim": 64,
            "noise_dim": 100,
            "color_mode": 1,
            "sparse": False
        }
    
    @pytest.fixture
    def test_input(self, default_params):
        """Test input tensor for discriminator."""
        batch_size = 2
        channels = 6 if default_params["color_mode"] == 1 else 3
        return torch.randn(
            batch_size,
            default_params["void_dim"],
            default_params["void_dim"],
            default_params["void_dim"],
            channels
        )
    
    def test_simple_discriminator_creation(self, default_params):
        """Test SimpleDiscriminator creation and functionality."""
        discriminator = SimpleDiscriminator(**default_params)
        assert isinstance(discriminator, nn.Module)
    
    def test_simple_discriminator_forward(self, default_params, test_input):
        """Test SimpleDiscriminator forward pass."""
        discriminator = SimpleDiscriminator(**default_params)
        output = discriminator(test_input)
        
        expected_shape = (test_input.size(0), 1)
        assert output.shape == expected_shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        
        # Check output is in valid range for sigmoid
        assert torch.all(output >= 0) and torch.all(output <= 1)
    
    def test_complex_discriminator_creation(self, default_params, test_input):
        """Test ComplexDiscriminator creation and functionality."""
        discriminator = ComplexDiscriminator(**default_params)
        assert isinstance(discriminator, nn.Module)
        
        output = discriminator(test_input)
        expected_shape = (test_input.size(0), 1)
        assert output.shape == expected_shape
    
    def test_spectral_norm_discriminator_creation(self, default_params, test_input):
        """Test SpectralNormDiscriminator creation and functionality."""
        discriminator = SpectralNormDiscriminator(**default_params)
        assert isinstance(discriminator, nn.Module)
        
        # Check that spectral normalization is applied
        has_spectral_norm = any(hasattr(module, 'weight_u') for module in discriminator.modules())
        assert has_spectral_norm
        
        output = discriminator(test_input)
        expected_shape = (test_input.size(0), 1)
        assert output.shape == expected_shape
    
    def test_progressive_discriminator_creation(self, default_params, test_input):
        """Test ProgressiveDiscriminator creation and functionality."""
        discriminator = ProgressiveDiscriminator(**default_params, max_resolution=128)
        assert isinstance(discriminator, nn.Module)
        
        output = discriminator(test_input)
        expected_shape = (test_input.size(0), 1)
        assert output.shape == expected_shape
        
        # Test growing functionality
        initial_level = discriminator.current_level
        discriminator.grow()
        assert discriminator.current_level == initial_level + 1
    
    def test_multiscale_discriminator_creation(self, default_params, test_input):
        """Test MultiScaleDiscriminator creation and functionality."""
        num_scales = 3
        discriminator = MultiScaleDiscriminator(**default_params, num_scales=num_scales)
        assert isinstance(discriminator, nn.Module)
        
        outputs = discriminator(test_input)
        assert isinstance(outputs, list)
        assert len(outputs) == num_scales
        
        for output in outputs:
            assert output.shape[0] == test_input.size(0)
            assert output.shape[1] == 1
    
    def test_conditional_discriminator_creation(self, default_params, test_input):
        """Test ConditionalDiscriminator creation and functionality."""
        condition_dim = 10
        discriminator = ConditionalDiscriminator(**default_params, condition_dim=condition_dim)
        assert isinstance(discriminator, nn.Module)
        
        condition = torch.randint(0, condition_dim, (test_input.size(0),))
        output = discriminator(test_input, condition)
        
        expected_shape = (test_input.size(0), 1)
        assert output.shape == expected_shape


class TestModelFactory:
    """Test PyTorchModelFactory functionality."""
    
    def test_factory_generator_creation(self):
        """Test factory generator creation."""
        generator = PyTorchModelFactory.create_generator("simple", device="cpu")
        assert isinstance(generator, nn.Module)
        
        # Test with different model types
        for model_type in ["simple", "complex", "skip", "monochrome", "autoencoder"]:
            gen = PyTorchModelFactory.create_generator(model_type, device="cpu")
            assert isinstance(gen, nn.Module)
    
    def test_factory_discriminator_creation(self):
        """Test factory discriminator creation."""
        discriminator = PyTorchModelFactory.create_discriminator("simple", device="cpu")
        assert isinstance(discriminator, nn.Module)
        
        # Test with different model types
        for model_type in ["simple", "complex", "skip", "monochrome", "autoencoder"]:
            disc = PyTorchModelFactory.create_discriminator(model_type, device="cpu")
            assert isinstance(disc, nn.Module)
    
    def test_factory_gan_pair_creation(self):
        """Test factory GAN pair creation."""
        generator, discriminator = PyTorchModelFactory.create_gan_pair("skip", device="cpu")
        
        assert isinstance(generator, nn.Module)
        assert isinstance(discriminator, nn.Module)
        
        # Test that they work together
        noise = torch.randn(1, 100)
        fake_data = generator(noise)
        prediction = discriminator(fake_data)
        
        assert prediction.shape == (1, 1)
    
    def test_factory_model_info(self):
        """Test factory model info functionality."""
        info = PyTorchModelFactory.get_model_info("skip")
        assert isinstance(info, dict)
        assert "description" in info
        assert "complexity" in info
    
    def test_factory_model_recommendation(self):
        """Test factory model recommendation."""
        # Test different scenarios
        data_chars = {"resolution": 64, "complexity": "medium", "memory_constraints": False}
        recommendation = PyTorchModelFactory.recommend_model(data_chars)
        assert recommendation in PyTorchModelFactory.GENERATOR_REGISTRY
        
        # Test memory constrained
        data_chars = {"resolution": 64, "complexity": "low", "memory_constraints": True}
        recommendation = PyTorchModelFactory.recommend_model(data_chars)
        assert recommendation in ["simple", "monochrome"]
        
        # Test conditional requirement
        data_chars = {"requires_conditioning": True}
        recommendation = PyTorchModelFactory.recommend_model(data_chars)
        assert recommendation == "conditional"
    
    def test_factory_list_models(self):
        """Test factory model listing."""
        models = PyTorchModelFactory.list_available_models()
        assert isinstance(models, dict)
        assert "generators" in models
        assert "discriminators" in models
        assert len(models["generators"]) > 0
        assert len(models["discriminators"]) > 0
    
    def test_factory_config_validation(self):
        """Test factory configuration validation."""
        # Valid config
        valid_config = {
            "model_type": "skip",
            "void_dim": 64,
            "noise_dim": 100,
            "color_mode": 1
        }
        is_valid, message = PyTorchModelFactory.validate_model_config(valid_config)
        assert is_valid
        assert "valid" in message.lower()
        
        # Invalid config - missing parameter
        invalid_config = {
            "model_type": "skip",
            "void_dim": 64,
            "noise_dim": 100
            # Missing color_mode
        }
        is_valid, message = PyTorchModelFactory.validate_model_config(invalid_config)
        assert not is_valid
        assert "missing" in message.lower()
        
        # Invalid config - wrong value
        invalid_config = {
            "model_type": "skip",
            "void_dim": 63,  # Invalid dimension
            "noise_dim": 100,
            "color_mode": 1
        }
        is_valid, message = PyTorchModelFactory.validate_model_config(invalid_config)
        assert not is_valid


class TestSparseOperations:
    """Test sparse tensor operations."""
    
    def test_sparse_conv3d_creation(self):
        """Test SparseConv3d layer creation."""
        layer = SparseConv3d(16, 32, 3, 1, 1)
        assert isinstance(layer, nn.Module)
    
    def test_sparse_conv3d_forward_dense(self):
        """Test SparseConv3d forward pass with dense tensor."""
        layer = SparseConv3d(16, 32, 3, 1, 1)
        input_tensor = torch.randn(2, 16, 8, 8, 8)
        output = layer(input_tensor)
        
        assert output.shape == (2, 32, 8, 8, 8)
        assert not output.is_sparse
    
    def test_sparse_conv3d_forward_sparse(self):
        """Test SparseConv3d forward pass with sparse tensor."""
        layer = SparseConv3d(16, 32, 3, 1, 1)
        
        # Create sparse tensor
        dense_tensor = torch.randn(2, 16, 8, 8, 8)
        dense_tensor[dense_tensor < 0.5] = 0  # Make it sparse
        sparse_tensor = dense_tensor.to_sparse()
        
        output = layer(sparse_tensor)
        assert output.shape == (2, 32, 8, 8, 8)
    
    def test_sparse_conv_transpose3d_creation(self):
        """Test SparseConvTranspose3d layer creation."""
        layer = SparseConvTranspose3d(16, 32, 3, 2, 1, 1)
        assert isinstance(layer, nn.Module)
    
    def test_sparse_batch_norm3d_creation(self):
        """Test SparseBatchNorm3d layer creation."""
        layer = SparseBatchNorm3d(16)
        assert isinstance(layer, nn.Module)


class TestModelUtils:
    """Test model utility functions."""
    
    def test_count_parameters(self):
        """Test parameter counting."""
        model = SimpleGenerator(void_dim=32, noise_dim=50, device="cpu")
        param_count = ModelUtils.count_parameters(model)
        
        assert isinstance(param_count, int)
        assert param_count > 0
    
    def test_get_model_size(self):
        """Test model size calculation."""
        model = SimpleGenerator(void_dim=32, noise_dim=50, device="cpu")
        size_mb = ModelUtils.get_model_size(model)
        
        assert isinstance(size_mb, float)
        assert size_mb > 0
    
    def test_estimate_memory_usage(self):
        """Test memory usage estimation."""
        model = SimpleGenerator(void_dim=32, noise_dim=50, device="cpu")
        input_shape = (50,)
        
        memory_usage = ModelUtils.estimate_memory_usage(model, input_shape, device="cpu")
        
        assert isinstance(memory_usage, dict)
        assert "model_size_mb" in memory_usage
        assert "activation_memory_mb" in memory_usage
        assert "total_memory_mb" in memory_usage
    
    def test_compare_models(self):
        """Test model comparison."""
        model1 = SimpleGenerator(void_dim=32, noise_dim=50, device="cpu")
        model2 = ComplexGenerator(void_dim=32, noise_dim=50, device="cpu")
        
        comparison = ModelUtils.compare_models(model1, model2)
        
        assert isinstance(comparison, dict)
        assert "model1_params" in comparison
        assert "model2_params" in comparison
        assert "param_ratio" in comparison
        assert "model1_size_mb" in comparison
        assert "model2_size_mb" in comparison
        assert "size_ratio" in comparison


class TestBackwardCompatibility:
    """Test backward compatibility functions."""
    
    def test_backward_compatible_generator_creation(self):
        """Test backward compatible generator creation."""
        generator = create_generator("simple", device="cpu")
        assert isinstance(generator, nn.Module)
    
    def test_backward_compatible_discriminator_creation(self):
        """Test backward compatible discriminator creation."""
        discriminator = create_discriminator("simple", device="cpu")
        assert isinstance(discriminator, nn.Module)


@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not available")
class TestTensorFlowEquivalence:
    """Test equivalence with TensorFlow models."""
    
    def test_generator_output_shapes_match(self):
        """Test that PyTorch and TensorFlow generators produce same output shapes."""
        # Create TensorFlow model
        tf_generator = TFModelFactory.create_generator("simple", void_dim=64, noise_dim=100, color_mode=1)
        
        # Create PyTorch model
        pytorch_generator = PyTorchModelFactory.create_generator("simple", void_dim=64, noise_dim=100, color_mode=1, device="cpu")
        
        # Test input
        noise_np = np.random.randn(2, 100).astype(np.float32)
        noise_torch = torch.from_numpy(noise_np)
        
        # Get outputs
        tf_output = tf_generator(noise_np)
        pytorch_output = pytorch_generator(noise_torch)
        
        # Compare shapes
        assert tf_output.shape == pytorch_output.shape
    
    def test_discriminator_output_shapes_match(self):
        """Test that PyTorch and TensorFlow discriminators produce same output shapes."""
        # Create TensorFlow model
        tf_discriminator = TFModelFactory.create_discriminator("simple", void_dim=64, noise_dim=100, color_mode=1)
        
        # Create PyTorch model
        pytorch_discriminator = PyTorchModelFactory.create_discriminator("simple", void_dim=64, noise_dim=100, color_mode=1, device="cpu")
        
        # Test input
        input_shape = (2, 64, 64, 64, 6)
        input_np = np.random.randn(*input_shape).astype(np.float32)
        input_torch = torch.from_numpy(input_np)
        
        # Get outputs
        tf_output = tf_discriminator(input_np)
        pytorch_output = pytorch_discriminator(input_torch)
        
        # Compare shapes
        assert tf_output.shape == pytorch_output.shape


class TestModelArchitectureSearch:
    """Test model architecture search functionality."""
    
    def test_architecture_search_creation(self):
        """Test architecture search creation."""
        search_space = {
            "model_type": ["simple", "skip", "complex"],
            "void_dim": [32, 64],
            "noise_dim": {"min": 50, "max": 150},
            "color_mode": [0, 1]
        }
        
        search = ModelArchitectureSearch(search_space)
        assert isinstance(search, ModelArchitectureSearch)
        assert search.search_space == search_space
    
    def test_architecture_search_run(self):
        """Test running architecture search."""
        search_space = {
            "model_type": ["simple", "skip"],
            "void_dim": [32, 64],
            "noise_dim": {"min": 50, "max": 100},
            "color_mode": [0, 1]
        }
        
        search = ModelArchitectureSearch(search_space)
        best_config = search.search(num_trials=3)
        
        assert isinstance(best_config, dict)
        assert "model_type" in best_config
        assert "void_dim" in best_config
        assert "noise_dim" in best_config
        assert "color_mode" in best_config
        
        # Check search history
        assert len(search.search_history) == 3
        for entry in search.search_history:
            assert "trial" in entry
            assert "config" in entry
            assert "score" in entry


if __name__ == "__main__":
    pytest.main([__file__])