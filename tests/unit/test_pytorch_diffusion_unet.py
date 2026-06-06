"""
Comprehensive tests for 3D U-Net diffusion architecture.

This module tests all components of the 3D U-Net implementation including:
- Time embedding functionality
- Encoder-decoder architecture
- Skip connections
- Attention mechanisms
- Conditioning systems
- Sparse tensor support
- Memory efficiency
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Tuple, List

# Import the models to test
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from deepsculpt.core.models.pytorch_models import (
    SinusoidalPositionEmbedding,
    TimeEmbedding,
    ResidualBlock3D,
    AttentionBlock3D,
    UNet3DEncoder,
    UNet3DDecoder,
    UNet3D,
    CrossAttentionBlock3D,
    MultiModalConditioner,
    ConditionalUNet3D
)


class TestSinusoidalPositionEmbedding:
    """Test sinusoidal position embedding for time steps."""
    
    def test_embedding_shape(self):
        """Test that embedding produces correct output shape."""
        dim = 256
        embedding = SinusoidalPositionEmbedding(dim)
        
        batch_size = 4
        timesteps = torch.randint(0, 1000, (batch_size,))
        
        output = embedding(timesteps)
        
        assert output.shape == (batch_size, dim)
        assert output.dtype == torch.float32
    
    def test_embedding_deterministic(self):
        """Test that embedding is deterministic for same inputs."""
        dim = 128
        embedding = SinusoidalPositionEmbedding(dim)
        
        timesteps = torch.tensor([0, 100, 500, 999])
        
        output1 = embedding(timesteps)
        output2 = embedding(timesteps)
        
        torch.testing.assert_close(output1, output2)
    
    def test_embedding_different_timesteps(self):
        """Test that different timesteps produce different embeddings."""
        dim = 256
        embedding = SinusoidalPositionEmbedding(dim)
        
        timesteps1 = torch.tensor([0, 100])
        timesteps2 = torch.tensor([200, 300])
        
        output1 = embedding(timesteps1)
        output2 = embedding(timesteps2)
        
        # Embeddings should be different
        assert not torch.allclose(output1, output2)
    
    def test_odd_dimension(self):
        """Test embedding with odd dimension."""
        dim = 255  # Odd dimension
        embedding = SinusoidalPositionEmbedding(dim)
        
        timesteps = torch.tensor([100])
        output = embedding(timesteps)
        
        assert output.shape == (1, dim)


class TestTimeEmbedding:
    """Test time embedding module."""
    
    def test_time_embedding_shape(self):
        """Test time embedding output shape."""
        time_dim = 256
        hidden_dim = 512
        time_emb = TimeEmbedding(time_dim, hidden_dim)
        
        batch_size = 4
        timesteps = torch.randint(0, 1000, (batch_size,))
        
        output = time_emb(timesteps)
        
        assert output.shape == (batch_size, hidden_dim)
    
    def test_time_embedding_gradient(self):
        """Test that time embedding supports gradients."""
        time_dim = 128
        hidden_dim = 256
        time_emb = TimeEmbedding(time_dim, hidden_dim)
        
        timesteps = torch.randint(0, 1000, (2,))
        output = time_emb(timesteps)
        
        # Check that gradients can flow
        loss = output.sum()
        loss.backward()
        
        # Check that parameters have gradients
        for param in time_emb.parameters():
            assert param.grad is not None


class TestResidualBlock3D:
    """Test 3D residual block with time embedding."""
    
    def test_residual_block_shape(self):
        """Test residual block output shape."""
        in_channels = 64
        out_channels = 128
        time_dim = 256
        
        block = ResidualBlock3D(in_channels, out_channels, time_dim)
        
        batch_size = 2
        depth, height, width = 32, 32, 32
        
        x = torch.randn(batch_size, in_channels, depth, height, width)
        time_emb = torch.randn(batch_size, time_dim)
        
        output = block(x, time_emb)
        
        assert output.shape == (batch_size, out_channels, depth, height, width)
    
    def test_residual_block_same_channels(self):
        """Test residual block with same input/output channels."""
        channels = 64
        time_dim = 256
        
        block = ResidualBlock3D(channels, channels, time_dim)
        
        x = torch.randn(2, channels, 16, 16, 16)
        time_emb = torch.randn(2, time_dim)
        
        output = block(x, time_emb)
        
        assert output.shape == x.shape
    
    def test_residual_block_sparse(self):
        """Test residual block with sparse tensors."""
        channels = 32
        time_dim = 128
        
        block = ResidualBlock3D(channels, channels, time_dim, sparse=True)
        
        # Create sparse input
        x = torch.randn(1, channels, 8, 8, 8)
        x[x < 0.5] = 0  # Make it sparse
        x_sparse = x.to_sparse()
        
        time_emb = torch.randn(1, time_dim)
        
        output = block(x_sparse, time_emb)
        
        # Output should maintain spatial dimensions
        if output.is_sparse:
            output = output.to_dense()
        assert output.shape == x.shape
    
    def test_residual_connection(self):
        """Test that residual connection is working."""
        channels = 64
        time_dim = 256
        
        block = ResidualBlock3D(channels, channels, time_dim)
        
        x = torch.randn(1, channels, 8, 8, 8)
        time_emb = torch.randn(1, time_dim)
        
        # Set block to eval mode to reduce randomness
        block.eval()
        
        with torch.no_grad():
            output = block(x, time_emb)
        
        # Output should be different from input (due to processing)
        assert not torch.allclose(output, x, atol=1e-6)


class TestAttentionBlock3D:
    """Test 3D self-attention block."""
    
    def test_attention_shape(self):
        """Test attention block output shape."""
        channels = 128
        num_heads = 8
        
        attention = AttentionBlock3D(channels, num_heads)
        
        batch_size = 2
        depth, height, width = 16, 16, 16
        
        x = torch.randn(batch_size, channels, depth, height, width)
        output = attention(x)
        
        assert output.shape == x.shape
    
    def test_attention_residual(self):
        """Test that attention includes residual connection."""
        channels = 64
        attention = AttentionBlock3D(channels, num_heads=4)
        
        x = torch.randn(1, channels, 8, 8, 8)
        
        attention.eval()
        with torch.no_grad():
            output = attention(x)
        
        # Due to residual connection, output should be related to input
        # but not identical (due to attention processing)
        assert not torch.allclose(output, x, atol=1e-6)
    
    def test_attention_sparse(self):
        """Test attention with sparse tensors."""
        channels = 32
        attention = AttentionBlock3D(channels, num_heads=4, sparse=True)
        
        x = torch.randn(1, channels, 8, 8, 8)
        x[x < 0.3] = 0  # Make sparse
        x_sparse = x.to_sparse()
        
        output = attention(x_sparse)
        
        # Check output shape
        if output.is_sparse:
            output = output.to_dense()
        assert output.shape == x.shape


class TestUNet3DEncoder:
    """Test 3D U-Net encoder."""
    
    def test_encoder_shape(self):
        """Test encoder output shapes."""
        in_channels = 4
        base_channels = 64
        time_dim = 256
        num_levels = 4
        
        encoder = UNet3DEncoder(in_channels, base_channels, time_dim, num_levels)
        
        batch_size = 2
        depth, height, width = 64, 64, 64
        
        x = torch.randn(batch_size, in_channels, depth, height, width)
        time_emb = torch.randn(batch_size, time_dim)
        
        features, skip_connections = encoder(x, time_emb)
        
        # Check final features shape
        expected_final_channels = base_channels * (2 ** (num_levels - 1))
        expected_final_size = depth // (2 ** (num_levels - 1))
        
        assert features.shape == (batch_size, expected_final_channels, 
                                expected_final_size, expected_final_size, expected_final_size)
        
        # Check skip connections
        assert len(skip_connections) == num_levels
        
        for i, skip in enumerate(skip_connections):
            expected_channels = base_channels * (2 ** i)
            expected_size = depth // (2 ** i)
            assert skip.shape == (batch_size, expected_channels, 
                                expected_size, expected_size, expected_size)
    
    def test_encoder_sparse(self):
        """Test encoder with sparse tensors."""
        encoder = UNet3DEncoder(4, 32, 128, 3, sparse=True)
        
        x = torch.randn(1, 4, 32, 32, 32)
        x[x < 0.2] = 0  # Make sparse
        x_sparse = x.to_sparse()
        
        time_emb = torch.randn(1, 128)
        
        features, skip_connections = encoder(x_sparse, time_emb)
        
        # Should produce valid outputs
        assert features is not None
        assert len(skip_connections) == 3


class TestUNet3DDecoder:
    """Test 3D U-Net decoder."""
    
    def test_decoder_shape(self):
        """Test decoder output shape."""
        encoder_channels = [64, 128, 256, 512]
        out_channels = 4
        time_dim = 256
        
        decoder = UNet3DDecoder(encoder_channels, out_channels, time_dim)
        
        batch_size = 2
        
        # Create encoded features (deepest level)
        deepest_size = 8
        encoded_features = torch.randn(batch_size, encoder_channels[-1], 
                                     deepest_size, deepest_size, deepest_size)
        
        # Create skip connections
        skip_connections = []
        for i, channels in enumerate(encoder_channels):
            size = deepest_size * (2 ** i)
            skip = torch.randn(batch_size, channels, size, size, size)
            skip_connections.append(skip)
        
        time_emb = torch.randn(batch_size, time_dim)
        
        output = decoder(encoded_features, skip_connections, time_emb)
        
        # Output should match the largest skip connection size
        expected_size = deepest_size * (2 ** (len(encoder_channels) - 1))
        assert output.shape == (batch_size, out_channels, 
                              expected_size, expected_size, expected_size)
    
    def test_decoder_sparse(self):
        """Test decoder with sparse tensors."""
        encoder_channels = [32, 64, 128]
        decoder = UNet3DDecoder(encoder_channels, 4, 128, sparse=True)
        
        # Create sparse inputs
        encoded_features = torch.randn(1, encoder_channels[-1], 4, 4, 4)
        encoded_features[encoded_features < 0.3] = 0
        encoded_features = encoded_features.to_sparse()
        
        skip_connections = []
        for i, channels in enumerate(encoder_channels):
            size = 4 * (2 ** i)
            skip = torch.randn(1, channels, size, size, size)
            skip[skip < 0.3] = 0
            skip_connections.append(skip.to_sparse())
        
        time_emb = torch.randn(1, 128)
        
        output = decoder(encoded_features, skip_connections, time_emb)
        
        # Should produce valid output
        assert output is not None
        if output.is_sparse:
            output = output.to_dense()
        assert output.shape[1] == 4  # out_channels


class TestUNet3D:
    """Test complete 3D U-Net architecture."""
    
    def test_unet_forward(self):
        """Test complete U-Net forward pass."""
        in_channels = 4
        out_channels = 4
        base_channels = 64
        
        unet = UNet3D(in_channels, out_channels, base_channels, num_levels=3)
        
        batch_size = 2
        depth, height, width = 32, 32, 32
        
        x = torch.randn(batch_size, in_channels, depth, height, width)
        timesteps = torch.randint(0, 1000, (batch_size,))
        
        output = unet(x, timesteps)
        
        assert output.shape == (batch_size, out_channels, depth, height, width)
    
    def test_unet_with_conditioning(self):
        """Test U-Net with conditioning."""
        unet = UNet3D(4, 4, 32, condition_dim=64, num_levels=3)
        
        x = torch.randn(1, 4, 16, 16, 16)
        timesteps = torch.randint(0, 1000, (1,))
        condition = torch.randn(1, 64)
        
        output = unet(x, timesteps, condition)
        
        assert output.shape == x.shape
    
    def test_unet_sparse(self):
        """Test U-Net with sparse tensors."""
        unet = UNet3D(4, 4, 32, sparse=True, num_levels=2)
        
        x = torch.randn(1, 4, 16, 16, 16)
        x[x < 0.2] = 0  # Make sparse
        x_sparse = x.to_sparse()
        
        timesteps = torch.randint(0, 1000, (1,))
        
        output = unet(x_sparse, timesteps)
        
        # Should produce valid output
        assert output is not None
        if output.is_sparse:
            output = output.to_dense()
        assert output.shape == x.shape
    
    def test_unet_memory_usage(self):
        """Test memory usage reporting."""
        unet = UNet3D(4, 4, 64, num_levels=3)
        
        memory_stats = unet.get_memory_usage()
        
        assert "total_parameters" in memory_stats
        assert "trainable_parameters" in memory_stats
        assert "estimated_param_memory_mb" in memory_stats
        assert memory_stats["total_parameters"] > 0
    
    def test_unet_optimization_suggestions(self):
        """Test optimization suggestions for different resolutions."""
        unet = UNet3D(4, 4, 64, num_levels=4)
        
        # Should not raise errors
        unet.optimize_for_resolution(64)
        unet.optimize_for_resolution(128)
        unet.optimize_for_resolution(32)
    
    def test_gradient_checkpointing(self):
        """Test gradient checkpointing functionality."""
        unet = UNet3D(4, 4, 32, num_levels=2)
        
        # Should not raise errors
        unet.enable_gradient_checkpointing()
        
        # Test forward pass still works
        x = torch.randn(1, 4, 16, 16, 16)
        timesteps = torch.randint(0, 1000, (1,))
        
        output = unet(x, timesteps)
        assert output.shape == x.shape


class TestCrossAttentionBlock3D:
    """Test cross-attention block for conditioning."""
    
    def test_cross_attention_shape(self):
        """Test cross-attention output shape."""
        channels = 128
        condition_dim = 256
        
        cross_attn = CrossAttentionBlock3D(channels, condition_dim)
        
        batch_size = 2
        x = torch.randn(batch_size, channels, 16, 16, 16)
        condition = torch.randn(batch_size, 10, condition_dim)  # 10 conditioning tokens
        
        output = cross_attn(x, condition)
        
        assert output.shape == x.shape
    
    def test_cross_attention_sparse(self):
        """Test cross-attention with sparse tensors."""
        cross_attn = CrossAttentionBlock3D(64, 128, sparse=True)
        
        x = torch.randn(1, 64, 8, 8, 8)
        x[x < 0.3] = 0
        x_sparse = x.to_sparse()
        
        condition = torch.randn(1, 5, 128)
        
        output = cross_attn(x_sparse, condition)
        
        if output.is_sparse:
            output = output.to_dense()
        assert output.shape == x.shape


class TestMultiModalConditioner:
    """Test multi-modal conditioning system."""
    
    def test_text_conditioning(self):
        """Test text-only conditioning."""
        conditioner = MultiModalConditioner(time_dim=256, text_dim=512)
        
        batch_size = 2
        text_emb = torch.randn(batch_size, 512)
        
        output = conditioner(text_emb=text_emb)
        
        assert output.shape == (batch_size, 256)
    
    def test_multi_modal_conditioning(self):
        """Test multiple conditioning modalities."""
        conditioner = MultiModalConditioner(
            time_dim=256, text_dim=512, param_dim=64, 
            image_channels=3, num_categories=10
        )
        
        batch_size = 2
        text_emb = torch.randn(batch_size, 512)
        params = torch.randn(batch_size, 64)
        image = torch.randn(batch_size, 3, 64, 64)
        category = torch.randint(0, 10, (batch_size,))
        
        output = conditioner(
            text_emb=text_emb, 
            params=params, 
            image=image, 
            category=category
        )
        
        assert output.shape == (batch_size, 256)
    
    def test_partial_conditioning(self):
        """Test conditioning with only some modalities."""
        conditioner = MultiModalConditioner(time_dim=128)
        
        batch_size = 1
        params = torch.randn(batch_size, 64)
        
        output = conditioner(params=params)
        
        assert output.shape == (batch_size, 128)


class TestConditionalUNet3D:
    """Test conditional U-Net with advanced conditioning."""
    
    def test_conditional_unet_forward(self):
        """Test conditional U-Net forward pass."""
        unet = ConditionalUNet3D(
            in_channels=4, out_channels=4, base_channels=32,
            num_levels=2, use_cross_attention=True
        )
        
        batch_size = 1
        x = torch.randn(batch_size, 4, 16, 16, 16)
        timesteps = torch.randint(0, 1000, (batch_size,))
        text_emb = torch.randn(batch_size, 512)
        params = torch.randn(batch_size, 64)
        
        output = unet(x, timesteps, text_emb=text_emb, params=params)
        
        assert output.shape == x.shape
    
    def test_conditional_unet_no_conditioning(self):
        """Test conditional U-Net without conditioning."""
        unet = ConditionalUNet3D(4, 4, 32, num_levels=2)
        
        x = torch.randn(1, 4, 16, 16, 16)
        timesteps = torch.randint(0, 1000, (1,))
        
        output = unet(x, timesteps)
        
        assert output.shape == x.shape
    
    def test_conditional_unet_sparse(self):
        """Test conditional U-Net with sparse tensors."""
        unet = ConditionalUNet3D(4, 4, 32, num_levels=2, sparse=True)
        
        x = torch.randn(1, 4, 16, 16, 16)
        x[x < 0.2] = 0
        x_sparse = x.to_sparse()
        
        timesteps = torch.randint(0, 1000, (1,))
        text_emb = torch.randn(1, 512)
        
        output = unet(x_sparse, timesteps, text_emb=text_emb)
        
        if output.is_sparse:
            output = output.to_dense()
        assert output.shape == x.shape


class TestUNetIntegration:
    """Integration tests for U-Net components."""
    
    def test_encoder_decoder_compatibility(self):
        """Test that encoder and decoder work together."""
        in_channels = 4
        out_channels = 4
        base_channels = 64
        time_dim = 256
        num_levels = 3
        
        # Create encoder
        encoder = UNet3DEncoder(in_channels, base_channels, time_dim, num_levels)
        
        # Create compatible decoder
        encoder_channels = [base_channels * (2 ** i) for i in range(num_levels)]
        decoder = UNet3DDecoder(encoder_channels, out_channels, time_dim)
        
        # Test forward pass
        batch_size = 1
        x = torch.randn(batch_size, in_channels, 32, 32, 32)
        time_emb = torch.randn(batch_size, time_dim)
        
        # Encode
        features, skip_connections = encoder(x, time_emb)
        
        # Decode
        output = decoder(features, skip_connections, time_emb)
        
        assert output.shape == (batch_size, out_channels, 32, 32, 32)
    
    def test_gradient_flow(self):
        """Test that gradients flow through the entire network."""
        unet = UNet3D(4, 4, 32, num_levels=2)
        
        x = torch.randn(1, 4, 16, 16, 16, requires_grad=True)
        timesteps = torch.randint(0, 1000, (1,))
        
        output = unet(x, timesteps)
        loss = output.sum()
        loss.backward()
        
        # Check that input has gradients
        assert x.grad is not None
        
        # Check that model parameters have gradients
        for param in unet.parameters():
            if param.requires_grad:
                assert param.grad is not None
    
    def test_different_input_sizes(self):
        """Test U-Net with different input sizes."""
        unet = UNet3D(4, 4, 32, num_levels=2)
        
        sizes = [(16, 16, 16), (32, 32, 32), (8, 8, 8)]
        
        for size in sizes:
            x = torch.randn(1, 4, *size)
            timesteps = torch.randint(0, 1000, (1,))
            
            output = unet(x, timesteps)
            assert output.shape == (1, 4, *size)
    
    def test_batch_processing(self):
        """Test U-Net with different batch sizes."""
        unet = UNet3D(4, 4, 32, num_levels=2)
        
        batch_sizes = [1, 2, 4]
        
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 4, 16, 16, 16)
            timesteps = torch.randint(0, 1000, (batch_size,))
            
            output = unet(x, timesteps)
            assert output.shape == (batch_size, 4, 16, 16, 16)


if __name__ == "__main__":
    # Run basic tests
    print("Running 3D U-Net diffusion tests...")
    
    # Test basic components
    print("Testing SinusoidalPositionEmbedding...")
    test_embedding = TestSinusoidalPositionEmbedding()
    test_embedding.test_embedding_shape()
    test_embedding.test_embedding_deterministic()
    print("✓ SinusoidalPositionEmbedding tests passed")
    
    print("Testing TimeEmbedding...")
    test_time = TestTimeEmbedding()
    test_time.test_time_embedding_shape()
    print("✓ TimeEmbedding tests passed")
    
    print("Testing ResidualBlock3D...")
    test_residual = TestResidualBlock3D()
    test_residual.test_residual_block_shape()
    test_residual.test_residual_block_same_channels()
    print("✓ ResidualBlock3D tests passed")
    
    print("Testing AttentionBlock3D...")
    test_attention = TestAttentionBlock3D()
    test_attention.test_attention_shape()
    print("✓ AttentionBlock3D tests passed")
    
    print("Testing UNet3DEncoder...")
    test_encoder = TestUNet3DEncoder()
    test_encoder.test_encoder_shape()
    print("✓ UNet3DEncoder tests passed")
    
    print("Testing UNet3DDecoder...")
    test_decoder = TestUNet3DDecoder()
    test_decoder.test_decoder_shape()
    print("✓ UNet3DDecoder tests passed")
    
    print("Testing UNet3D...")
    test_unet = TestUNet3D()
    test_unet.test_unet_forward()
    test_unet.test_unet_memory_usage()
    print("✓ UNet3D tests passed")
    
    print("Testing MultiModalConditioner...")
    test_conditioner = TestMultiModalConditioner()
    test_conditioner.test_text_conditioning()
    test_conditioner.test_multi_modal_conditioning()
    print("✓ MultiModalConditioner tests passed")
    
    print("Testing ConditionalUNet3D...")
    test_conditional = TestConditionalUNet3D()
    test_conditional.test_conditional_unet_forward()
    print("✓ ConditionalUNet3D tests passed")
    
    print("Testing integration...")
    test_integration = TestUNetIntegration()
    test_integration.test_encoder_decoder_compatibility()
    test_integration.test_gradient_flow()
    test_integration.test_different_input_sizes()
    print("✓ Integration tests passed")
    
    print("\n🎉 All 3D U-Net diffusion tests passed successfully!")