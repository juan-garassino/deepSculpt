"""
Integration tests for PyTorch models to verify complete functionality.
"""

import torch
import torch.nn as nn
import sys
import os

# Add the deepSculpt module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from deepsculpt.core.models.pytorch_models import (
    PyTorchModelFactory, ModelUtils, create_generator, create_discriminator
)


def test_complete_gan_workflow():
    """Test complete GAN workflow with PyTorch models."""
    print("Testing complete GAN workflow...")
    
    # Create generator and discriminator
    generator = PyTorchModelFactory.create_generator("skip", device="cpu")
    discriminator = PyTorchModelFactory.create_discriminator("skip", device="cpu")
    
    # Set to eval mode for testing
    generator.eval()
    discriminator.eval()
    
    # Test forward pass
    batch_size = 4
    noise_dim = 100
    
    # Generate fake data
    noise = torch.randn(batch_size, noise_dim)
    fake_data = generator(noise)
    
    # Discriminate fake data
    fake_predictions = discriminator(fake_data)
    
    # Create real data (random for testing)
    real_data = torch.randn_like(fake_data)
    real_predictions = discriminator(real_data)
    
    # Verify shapes
    assert fake_data.shape == (batch_size, 64, 64, 64, 6)
    assert fake_predictions.shape == (batch_size, 1)
    assert real_predictions.shape == (batch_size, 1)
    
    # Verify outputs are in valid ranges
    assert torch.all(fake_predictions >= 0) and torch.all(fake_predictions <= 1)
    assert torch.all(real_predictions >= 0) and torch.all(real_predictions <= 1)
    
    print("✓ Complete GAN workflow test passed")


def test_different_model_architectures():
    """Test different model architectures."""
    print("Testing different model architectures...")
    
    model_types = ["simple", "complex", "skip", "monochrome", "autoencoder"]
    
    for model_type in model_types:
        print(f"  Testing {model_type}...")
        
        # Create models
        generator = PyTorchModelFactory.create_generator(model_type, device="cpu")
        discriminator = PyTorchModelFactory.create_discriminator(model_type, device="cpu")
        
        generator.eval()
        discriminator.eval()
        
        # Test forward pass
        noise = torch.randn(2, 100)
        fake_data = generator(noise)
        prediction = discriminator(fake_data)
        
        # Verify basic properties
        assert len(fake_data.shape) == 5  # batch, depth, height, width, channels
        assert prediction.shape == (2, 1)
        assert not torch.isnan(fake_data).any()
        assert not torch.isnan(prediction).any()
        
        print(f"    ✓ {model_type} architecture works correctly")
    
    print("✓ All model architectures test passed")


def test_different_configurations():
    """Test different model configurations."""
    print("Testing different model configurations...")
    
    configurations = [
        {"void_dim": 32, "noise_dim": 50, "color_mode": 0},
        {"void_dim": 64, "noise_dim": 100, "color_mode": 1},
        {"void_dim": 128, "noise_dim": 200, "color_mode": 1},
    ]
    
    for i, config in enumerate(configurations):
        print(f"  Testing configuration {i+1}: {config}")
        
        # Create models with configuration
        generator = PyTorchModelFactory.create_generator("simple", **config, device="cpu")
        discriminator = PyTorchModelFactory.create_discriminator("simple", **config, device="cpu")
        
        generator.eval()
        discriminator.eval()
        
        # Test forward pass
        noise = torch.randn(2, config["noise_dim"])
        fake_data = generator(noise)
        prediction = discriminator(fake_data)
        
        # Verify shapes match configuration
        expected_channels = 6 if config["color_mode"] == 1 else 3
        expected_shape = (2, config["void_dim"], config["void_dim"], config["void_dim"], expected_channels)
        
        assert fake_data.shape == expected_shape
        assert prediction.shape == (2, 1)
        
        print(f"    ✓ Configuration {i+1} works correctly")
    
    print("✓ All configurations test passed")


def test_model_utilities():
    """Test model utility functions."""
    print("Testing model utilities...")
    
    # Create a test model
    model = PyTorchModelFactory.create_generator("simple", device="cpu")
    
    # Test parameter counting
    param_count = ModelUtils.count_parameters(model)
    assert isinstance(param_count, int)
    assert param_count > 0
    print(f"  Model has {param_count:,} parameters")
    
    # Test model size calculation
    model_size = ModelUtils.get_model_size(model)
    assert isinstance(model_size, float)
    assert model_size > 0
    print(f"  Model size: {model_size:.2f} MB")
    
    # Test memory usage estimation
    input_shape = (100,)
    memory_usage = ModelUtils.estimate_memory_usage(model, input_shape, device="cpu")
    assert isinstance(memory_usage, dict)
    assert "model_size_mb" in memory_usage
    assert "total_memory_mb" in memory_usage
    print(f"  Estimated memory usage: {memory_usage['total_memory_mb']:.2f} MB")
    
    # Test model comparison
    model2 = PyTorchModelFactory.create_generator("complex", device="cpu")
    comparison = ModelUtils.compare_models(model, model2)
    assert isinstance(comparison, dict)
    assert "param_ratio" in comparison
    print(f"  Parameter ratio (complex/simple): {comparison['param_ratio']:.2f}")
    
    print("✓ Model utilities test passed")


def test_factory_functionality():
    """Test factory functionality."""
    print("Testing factory functionality...")
    
    # Test model creation
    generator, discriminator = PyTorchModelFactory.create_gan_pair("skip", device="cpu")
    assert isinstance(generator, nn.Module)
    assert isinstance(discriminator, nn.Module)
    print("  ✓ GAN pair creation works")
    
    # Test model info
    info = PyTorchModelFactory.get_model_info("skip")
    assert isinstance(info, dict)
    assert "description" in info
    print(f"  ✓ Model info: {info['description']}")
    
    # Test model recommendation
    data_chars = {"resolution": 64, "complexity": "medium"}
    recommendation = PyTorchModelFactory.recommend_model(data_chars)
    assert recommendation in PyTorchModelFactory.GENERATOR_REGISTRY
    print(f"  ✓ Model recommendation: {recommendation}")
    
    # Test model listing
    models = PyTorchModelFactory.list_available_models()
    assert "generators" in models
    assert "discriminators" in models
    print(f"  ✓ Available models: {len(models['generators'])} generators, {len(models['discriminators'])} discriminators")
    
    # Test config validation
    valid_config = {"model_type": "skip", "void_dim": 64, "noise_dim": 100, "color_mode": 1}
    is_valid, message = PyTorchModelFactory.validate_model_config(valid_config)
    assert is_valid
    print("  ✓ Config validation works")
    
    print("✓ Factory functionality test passed")


def test_backward_compatibility():
    """Test backward compatibility functions."""
    print("Testing backward compatibility...")
    
    # Test backward compatible functions
    generator = create_generator("simple", device="cpu")
    discriminator = create_discriminator("simple", device="cpu")
    
    assert isinstance(generator, nn.Module)
    assert isinstance(discriminator, nn.Module)
    
    # Test they work together
    generator.eval()
    discriminator.eval()
    
    noise = torch.randn(2, 100)
    fake_data = generator(noise)
    prediction = discriminator(fake_data)
    
    assert fake_data.shape == (2, 64, 64, 64, 6)
    assert prediction.shape == (2, 1)
    
    print("✓ Backward compatibility test passed")


def test_sparse_functionality():
    """Test sparse tensor functionality."""
    print("Testing sparse functionality...")
    
    # Create sparse and dense models
    gen_sparse = PyTorchModelFactory.create_generator("simple", sparse=True, device="cpu")
    gen_dense = PyTorchModelFactory.create_generator("simple", sparse=False, device="cpu")
    
    gen_sparse.eval()
    gen_dense.eval()
    
    # Test forward pass
    noise = torch.randn(2, 100)
    output_sparse = gen_sparse(noise)
    output_dense = gen_dense(noise)
    
    # Both should produce valid outputs
    assert output_sparse.shape == output_dense.shape
    assert not torch.isnan(output_sparse).any()
    assert not torch.isnan(output_dense).any()
    
    print("✓ Sparse functionality test passed")


def run_all_tests():
    """Run all integration tests."""
    print("=" * 60)
    print("Running PyTorch Models Integration Tests")
    print("=" * 60)
    
    try:
        test_complete_gan_workflow()
        test_different_model_architectures()
        test_different_configurations()
        test_model_utilities()
        test_factory_functionality()
        test_backward_compatibility()
        test_sparse_functionality()
        
        print("=" * 60)
        print("✅ ALL INTEGRATION TESTS PASSED!")
        print("=" * 60)
        
    except Exception as e:
        print("=" * 60)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 60)
        raise


if __name__ == "__main__":
    run_all_tests()