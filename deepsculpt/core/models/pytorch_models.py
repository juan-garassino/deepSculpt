"""
PyTorch model definitions for DeepSculpt.
This file contains all PyTorch model architectures equivalent to the TensorFlow versions.

LEGACY — parallel implementation superseded by gan/{generator,discriminator}.py,
diffusion/{unet,noise_scheduler,pipeline}.py and model_factory.py. Only the
guarded legacy workflow import and old unit tests reference it. Do not import
from here in live code (its classes shadow the canonical ones).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List, Union
import math


class SparseConv3d(nn.Module):
    """
    3D convolution layer that handles sparse tensors efficiently.
    
    This layer automatically detects sparse inputs and optimizes computation accordingly.
    It can maintain sparsity through the convolution operation when beneficial.
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1, bias: bool = True,
                 sparse_threshold: float = 0.1, auto_sparse: bool = True):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.sparse_threshold = sparse_threshold
        self.auto_sparse = auto_sparse
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Statistics for optimization decisions
        self.sparse_input_count = 0
        self.dense_input_count = 0
        self.total_forward_calls = 0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.total_forward_calls += 1
        
        if x.is_sparse:
            self.sparse_input_count += 1
            # Convert to dense for convolution
            x_dense = x.to_dense()
            out = self.conv(x_dense)
            
            # Decide whether to convert back to sparse
            if self.auto_sparse:
                sparsity = (out == 0).float().mean().item()
                if sparsity > self.sparse_threshold:
                    return out.to_sparse()
            return out
        else:
            self.dense_input_count += 1
            out = self.conv(x)
            
            # Optionally convert to sparse if output is very sparse
            if self.auto_sparse:
                sparsity = (out == 0).float().mean().item()
                if sparsity > self.sparse_threshold:
                    return out.to_sparse()
            return out
    
    def get_sparsity_stats(self) -> Dict[str, Any]:
        """Get statistics about sparsity patterns in this layer."""
        return {
            "total_forward_calls": self.total_forward_calls,
            "sparse_input_ratio": self.sparse_input_count / max(self.total_forward_calls, 1),
            "dense_input_ratio": self.dense_input_count / max(self.total_forward_calls, 1),
            "sparse_threshold": self.sparse_threshold,
            "auto_sparse_enabled": self.auto_sparse,
        }
    
    def optimize_for_sparsity(self):
        """Optimize layer parameters based on observed sparsity patterns."""
        stats = self.get_sparsity_stats()
        
        # Adjust threshold based on input patterns
        if stats["sparse_input_ratio"] > 0.8:
            # Mostly sparse inputs - be more aggressive about maintaining sparsity
            self.sparse_threshold = max(0.05, self.sparse_threshold * 0.8)
        elif stats["sparse_input_ratio"] < 0.2:
            # Mostly dense inputs - be less aggressive about sparsity
            self.sparse_threshold = min(0.5, self.sparse_threshold * 1.2)


class SparseConvTranspose3d(nn.Module):
    """
    3D transposed convolution for sparse tensors with adaptive sparsity management.
    
    This layer is particularly useful in generator networks where sparse representations
    can significantly reduce memory usage during upsampling operations.
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, output_padding: int = 0, bias: bool = True,
                 sparse_threshold: float = 0.1, auto_sparse: bool = True,
                 memory_efficient: bool = True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=bias
        )
        self.sparse_threshold = sparse_threshold
        self.auto_sparse = auto_sparse
        self.memory_efficient = memory_efficient
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Track memory usage patterns
        self.memory_savings_history = []
        self.sparsity_history = []
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_memory = self._estimate_memory_usage(x)
        
        if x.is_sparse:
            x_dense = x.to_dense()
            out = self.conv_transpose(x_dense)
            
            if self.auto_sparse:
                sparsity = (out == 0).float().mean().item()
                self.sparsity_history.append(sparsity)
                
                if sparsity > self.sparse_threshold:
                    sparse_out = out.to_sparse()
                    
                    if self.memory_efficient:
                        # Calculate memory savings
                        dense_memory = self._estimate_memory_usage(out)
                        sparse_memory = self._estimate_memory_usage(sparse_out)
                        memory_savings = (dense_memory - sparse_memory) / dense_memory
                        self.memory_savings_history.append(memory_savings)
                    
                    return sparse_out
            return out
        else:
            out = self.conv_transpose(x)
            
            if self.auto_sparse:
                sparsity = (out == 0).float().mean().item()
                self.sparsity_history.append(sparsity)
                
                if sparsity > self.sparse_threshold:
                    return out.to_sparse()
            return out
    
    def _estimate_memory_usage(self, tensor: torch.Tensor) -> int:
        """Estimate memory usage of a tensor in bytes."""
        if tensor.is_sparse:
            indices_memory = tensor._indices().element_size() * tensor._indices().numel()
            values_memory = tensor._values().element_size() * tensor._values().numel()
            return indices_memory + values_memory
        else:
            return tensor.element_size() * tensor.numel()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for this layer."""
        avg_sparsity = sum(self.sparsity_history) / len(self.sparsity_history) if self.sparsity_history else 0
        avg_memory_savings = sum(self.memory_savings_history) / len(self.memory_savings_history) if self.memory_savings_history else 0
        
        return {
            "average_output_sparsity": avg_sparsity,
            "average_memory_savings": avg_memory_savings,
            "total_forward_passes": len(self.sparsity_history),
            "sparse_threshold": self.sparse_threshold,
            "memory_efficient_mode": self.memory_efficient,
        }
    
    def adapt_threshold(self, target_memory_savings: float = 0.3):
        """Adapt the sparse threshold based on observed performance."""
        if len(self.memory_savings_history) < 10:
            return  # Need more data
        
        avg_savings = sum(self.memory_savings_history[-10:]) / 10
        
        if avg_savings < target_memory_savings:
            # Not saving enough memory, be more aggressive
            self.sparse_threshold = max(0.05, self.sparse_threshold * 0.9)
        elif avg_savings > target_memory_savings * 1.5:
            # Saving too much memory (might be losing quality), be less aggressive
            self.sparse_threshold = min(0.5, self.sparse_threshold * 1.1)


class SparseBatchNorm3d(nn.Module):
    """
    Batch normalization adapted for sparse tensors with efficient sparse-aware computation.
    
    This implementation maintains sparsity patterns while properly normalizing
    the non-zero elements of sparse tensors.
    """
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = True, track_running_stats: bool = True,
                 sparse_mode: str = "auto"):
        super().__init__()
        self.bn = nn.BatchNorm3d(num_features, eps=eps, momentum=momentum, 
                                affine=affine, track_running_stats=track_running_stats)
        self.sparse_mode = sparse_mode  # "auto", "preserve", "densify"
        self.num_features = num_features
        
        # Statistics for sparse processing
        self.sparse_processing_count = 0
        self.dense_processing_count = 0
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_sparse:
            self.sparse_processing_count += 1
            
            if self.sparse_mode == "preserve":
                # Preserve sparsity by only normalizing non-zero elements
                return self._sparse_batch_norm(x)
            else:
                # Convert to dense, normalize, then decide on output format
                x_dense = x.to_dense()
                out = self.bn(x_dense)
                
                if self.sparse_mode == "auto":
                    # Maintain sparsity if beneficial
                    sparsity = (out == 0).float().mean().item()
                    if sparsity > 0.1:  # Threshold for maintaining sparsity
                        return out.to_sparse()
                elif self.sparse_mode == "preserve":
                    return out.to_sparse()
                
                return out
        else:
            self.dense_processing_count += 1
            return self.bn(x)
    
    def _sparse_batch_norm(self, x: torch.sparse.FloatTensor) -> torch.sparse.FloatTensor:
        """
        Perform batch normalization on sparse tensor while preserving sparsity.
        
        This method normalizes only the non-zero values, which is more efficient
        for very sparse tensors but may not be mathematically equivalent to
        dense batch normalization.
        """
        # Get sparse tensor components
        indices = x._indices()
        values = x._values()
        shape = x.shape
        
        # Reshape values to match batch norm expectations
        # values shape: [nnz] -> [nnz, 1, 1, 1] for 3D batch norm
        if len(values.shape) == 1:
            # Add channel dimension if not present
            channel_indices = indices[1]  # Assuming indices are [batch, channel, d, h, w]
            
            # Group values by channel for normalization
            normalized_values = torch.zeros_like(values)
            
            for c in range(self.num_features):
                channel_mask = channel_indices == c
                if channel_mask.any():
                    channel_values = values[channel_mask]
                    
                    if self.training:
                        # Calculate statistics for this channel's non-zero values
                        mean = channel_values.mean()
                        var = channel_values.var(unbiased=False)
                        
                        # Update running statistics
                        if self.bn.track_running_stats:
                            with torch.no_grad():
                                self.bn.running_mean[c] = (1 - self.bn.momentum) * self.bn.running_mean[c] + self.bn.momentum * mean
                                self.bn.running_var[c] = (1 - self.bn.momentum) * self.bn.running_var[c] + self.bn.momentum * var
                    else:
                        # Use running statistics
                        mean = self.bn.running_mean[c]
                        var = self.bn.running_var[c]
                    
                    # Normalize
                    normalized_channel_values = (channel_values - mean) / torch.sqrt(var + self.bn.eps)
                    
                    # Apply affine transformation if enabled
                    if self.bn.affine:
                        normalized_channel_values = normalized_channel_values * self.bn.weight[c] + self.bn.bias[c]
                    
                    normalized_values[channel_mask] = normalized_channel_values
        else:
            # Values already have proper shape
            normalized_values = values
        
        # Create new sparse tensor with normalized values
        return torch.sparse.FloatTensor(indices, normalized_values, shape)
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get statistics about sparse vs dense processing."""
        total = self.sparse_processing_count + self.dense_processing_count
        return {
            "sparse_processing_ratio": self.sparse_processing_count / max(total, 1),
            "dense_processing_ratio": self.dense_processing_count / max(total, 1),
            "total_forward_passes": total,
            "sparse_mode": self.sparse_mode,
        }
    
    def set_sparse_mode(self, mode: str):
        """Set the sparse processing mode."""
        if mode not in ["auto", "preserve", "densify"]:
            raise ValueError("Mode must be one of: 'auto', 'preserve', 'densify'")
        self.sparse_mode = mode


def _final_activation(x: torch.Tensor, color_mode: int, output_channels: int) -> torch.Tensor:
    """Apply correct final activation based on output mode."""
    if color_mode == 1 and output_channels >= 6:
        return F.softmax(x, dim=1)  # OHE: pick one class
    elif output_channels == 3:
        return torch.tanh(x)  # RGB: continuous color [-1, 1]
    else:
        return torch.sigmoid(x)  # Mono: binary occupancy


class SimpleGenerator(nn.Module):
    """Simple generator model equivalent to TensorFlow version."""
    
    def __init__(self, void_dim: int = 64, noise_dim: int = 100, color_mode: int = 1, sparse: bool = False):
        super().__init__()
        self.void_dim = void_dim
        self.noise_dim = noise_dim
        self.color_mode = color_mode
        self.sparse = sparse
        
        # Output channels based on color mode
        self.output_channels = 6 if color_mode == 1 else 3
        
        # Initial dense layer
        self.initial_size = void_dim // 8
        self.fc = nn.Linear(noise_dim, self.initial_size ** 3 * noise_dim, bias=False)
        self.bn1 = nn.BatchNorm1d(self.initial_size ** 3 * noise_dim)
        
        # Transposed convolution layers
        ConvTranspose = SparseConvTranspose3d if sparse else nn.ConvTranspose3d
        BatchNorm = SparseBatchNorm3d if sparse else nn.BatchNorm3d
        
        self.conv1 = ConvTranspose(noise_dim, noise_dim, 3, 1, 1, bias=False)
        self.bn2 = BatchNorm(noise_dim)
        
        self.conv2 = ConvTranspose(noise_dim, noise_dim // 2, 3, 2, 1, 1, bias=False)
        self.bn3 = BatchNorm(noise_dim // 2)
        
        self.conv3 = ConvTranspose(noise_dim // 2, noise_dim // 4, 3, 2, 1, 1, bias=False)
        self.bn4 = BatchNorm(noise_dim // 4)
        
        self.conv4 = ConvTranspose(noise_dim // 4, self.output_channels, 3, 2, 1, 1, bias=False)
        
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial dense layer and reshape
        x = self.fc(x)
        # Handle single batch case for batch norm
        if x.size(0) == 1 and self.training:
            self.bn1.eval()
            x = self.bn1(x)
            self.bn1.train()
        else:
            x = self.bn1(x)
        x = self.relu(x)
        x = x.view(-1, self.noise_dim, self.initial_size, self.initial_size, self.initial_size)

        # First transposed conv block
        x = self.conv1(x)
        if x.size(0) == 1 and self.training:
            self.bn2.eval()
            x = self.bn2(x)
            self.bn2.train()
        else:
            x = self.bn2(x)
        x = self.relu(x)

        # Second transposed conv block
        x = self.conv2(x)
        if x.size(0) == 1 and self.training:
            self.bn3.eval()
            x = self.bn3(x)
            self.bn3.train()
        else:
            x = self.bn3(x)
        x = self.relu(x)

        # Third transposed conv block
        x = self.conv3(x)
        if x.size(0) == 1 and self.training:
            self.bn4.eval()
            x = self.bn4(x)
            self.bn4.train()
        else:
            x = self.bn4(x)
        x = self.relu(x)

        # Final transposed conv block
        x = self.conv4(x)
        x = _final_activation(x, self.color_mode, self.output_channels)
        
        # Reshape to final output
        x = x.view(-1, self.void_dim, self.void_dim, self.void_dim, self.output_channels)
        
        return x


class ComplexGenerator(nn.Module):
    """Complex generator model with skip connections."""
    
    def __init__(self, void_dim: int = 64, noise_dim: int = 100, color_mode: int = 1, sparse: bool = False):
        super().__init__()
        self.void_dim = void_dim
        self.noise_dim = noise_dim
        self.color_mode = color_mode
        self.sparse = sparse
        
        self.output_channels = 6 if color_mode == 1 else 3
        self.initial_size = void_dim // 8
        
        # Initial dense layer
        self.fc = nn.Linear(noise_dim, self.initial_size ** 3 * noise_dim, bias=False)
        self.bn1 = nn.BatchNorm1d(self.initial_size ** 3 * noise_dim)
        
        # Transposed convolution layers with skip connections
        ConvTranspose = SparseConvTranspose3d if sparse else nn.ConvTranspose3d
        BatchNorm = SparseBatchNorm3d if sparse else nn.BatchNorm3d
        
        self.conv1 = ConvTranspose(noise_dim, noise_dim, 3, 1, 1, bias=False)
        self.bn2 = BatchNorm(noise_dim)
        
        # Skip connection layer - concatenates with previous layer
        self.conv2 = ConvTranspose(noise_dim * 2, noise_dim // 2, 3, 2, 1, 1, bias=False)
        self.bn3 = BatchNorm(noise_dim // 2)
        
        self.conv3 = ConvTranspose(noise_dim // 2 * 2, noise_dim // 4, 3, 2, 1, 1, bias=False)
        self.bn4 = BatchNorm(noise_dim // 4)
        
        self.conv4 = ConvTranspose(noise_dim // 4 * 2, self.output_channels, 3, 2, 1, 1, bias=False)
        
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial dense layer and reshape
        x = self.fc(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = x.view(-1, self.noise_dim, self.initial_size, self.initial_size, self.initial_size)

        skip_connections = []

        # First layer
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        skip_connections.append(x)

        # Second layer with skip connection
        x = torch.cat([x, skip_connections[-1]], dim=1)
        x = self.conv2(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)
        skip_connections.append(x)

        # Third layer with skip connection
        x = torch.cat([x, skip_connections[-1]], dim=1)
        x = self.conv3(x)
        x = self.bn4(x)
        x = self.leaky_relu(x)
        skip_connections.append(x)

        # Final layer with skip connection
        x = torch.cat([x, skip_connections[-1]], dim=1)
        x = self.conv4(x)
        x = _final_activation(x, self.color_mode, self.output_channels)
        
        # Reshape to final output
        x = x.view(-1, self.void_dim, self.void_dim, self.void_dim, self.output_channels)
        
        return x


class SkipGenerator(nn.Module):
    """Generator with skip connections (U-Net style)."""
    
    def __init__(self, void_dim: int = 64, noise_dim: int = 100, color_mode: int = 1, sparse: bool = False):
        super().__init__()
        self.void_dim = void_dim
        self.noise_dim = noise_dim
        self.color_mode = color_mode
        self.sparse = sparse
        
        self.output_channels = 6 if color_mode == 1 else 3
        self.initial_size = void_dim // 8
        
        # Initial dense layer
        self.fc = nn.Linear(noise_dim, self.initial_size ** 3 * noise_dim, bias=False)
        self.bn1 = nn.BatchNorm1d(self.initial_size ** 3 * noise_dim)
        
        # Transposed convolution layers
        ConvTranspose = SparseConvTranspose3d if sparse else nn.ConvTranspose3d
        BatchNorm = SparseBatchNorm3d if sparse else nn.BatchNorm3d
        
        self.conv1 = ConvTranspose(noise_dim, noise_dim, 3, 1, 1, bias=False)
        self.bn2 = BatchNorm(noise_dim)
        
        self.conv2 = ConvTranspose(noise_dim * 2, noise_dim // 2, 3, 2, 1, 1, bias=False)
        self.bn3 = BatchNorm(noise_dim // 2)
        
        self.conv3 = ConvTranspose(noise_dim // 2 * 2, noise_dim // 4, 3, 2, 1, 1, bias=False)
        self.bn4 = BatchNorm(noise_dim // 4)
        
        self.conv4 = ConvTranspose(noise_dim // 4 * 2, self.output_channels, 3, 2, 1, 1, bias=False)
        
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial dense layer and reshape
        x = self.fc(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = x.view(-1, self.noise_dim, self.initial_size, self.initial_size, self.initial_size)

        skip_connections = []

        # First layer
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu(x)
        skip_connections.append(x)

        # Second layer with skip connection
        x = torch.cat([x, skip_connections[-1]], dim=1)
        x = self.conv2(x)
        x = self.bn3(x)
        x = self.relu(x)
        skip_connections.append(x)

        # Third layer with skip connection
        x = torch.cat([x, skip_connections[-1]], dim=1)
        x = self.conv3(x)
        x = self.bn4(x)
        x = self.relu(x)
        skip_connections.append(x)

        # Final layer with skip connection
        x = torch.cat([x, skip_connections[-1]], dim=1)
        x = self.conv4(x)
        x = _final_activation(x, self.color_mode, self.output_channels)
        
        # Reshape to final output
        x = x.view(-1, self.void_dim, self.void_dim, self.void_dim, self.output_channels)
        
        return x


class MonochromeGenerator(nn.Module):
    """Monochrome generator model."""
    
    def __init__(self, void_dim: int = 64, noise_dim: int = 100, color_mode: int = 0, sparse: bool = False):
        super().__init__()
        self.void_dim = void_dim
        self.noise_dim = noise_dim
        self.color_mode = color_mode
        self.sparse = sparse
        
        # Always 3 channels for monochrome
        self.output_channels = 3
        self.initial_size = void_dim // 8
        
        # Initial dense layer
        self.fc = nn.Linear(noise_dim, self.initial_size ** 3 * noise_dim, bias=False)
        self.bn1 = nn.BatchNorm1d(self.initial_size ** 3 * noise_dim)
        
        # Transposed convolution layers
        ConvTranspose = SparseConvTranspose3d if sparse else nn.ConvTranspose3d
        BatchNorm = SparseBatchNorm3d if sparse else nn.BatchNorm3d
        
        self.conv1 = ConvTranspose(noise_dim, noise_dim, 3, 1, 1, bias=False)
        self.bn2 = BatchNorm(noise_dim)
        
        self.conv2 = ConvTranspose(noise_dim, noise_dim // 2, 3, 2, 1, 1, bias=False)
        self.bn3 = BatchNorm(noise_dim // 2)
        
        self.conv3 = ConvTranspose(noise_dim // 2, noise_dim // 4, 3, 2, 1, 1, bias=False)
        self.bn4 = BatchNorm(noise_dim // 4)
        
        self.conv4 = ConvTranspose(noise_dim // 4, self.output_channels, 3, 2, 1, 1, bias=False)

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial dense layer and reshape
        x = self.fc(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = x.view(-1, self.noise_dim, self.initial_size, self.initial_size, self.initial_size)

        # First transposed conv block
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu(x)

        # Second transposed conv block
        x = self.conv2(x)
        x = self.bn3(x)
        x = self.relu(x)

        # Third transposed conv block
        x = self.conv3(x)
        x = self.bn4(x)
        x = self.relu(x)

        # Final transposed conv block
        x = self.conv4(x)
        x = _final_activation(x, self.color_mode, self.output_channels)
        
        # Reshape to final output
        x = x.view(-1, self.void_dim, self.void_dim, self.void_dim, self.output_channels)
        
        return x


class AutoencoderGenerator(nn.Module):
    """Generator based on autoencoder architecture."""
    
    def __init__(self, void_dim: int = 64, noise_dim: int = 100, color_mode: int = 1, sparse: bool = False):
        super().__init__()
        self.void_dim = void_dim
        self.noise_dim = noise_dim
        self.color_mode = color_mode
        self.sparse = sparse
        
        self.output_channels = 6 if color_mode == 1 else 3
        
        # Dense layer to expand the latent dimension
        self.fc = nn.Linear(noise_dim, 4 * 4 * 4 * 16)  # 1024 values for 4x4x4x16
        
        # Upsampling layers
        ConvTranspose = SparseConvTranspose3d if sparse else nn.ConvTranspose3d
        
        self.conv1 = ConvTranspose(16, 128, 5, 2, 2, 1)
        self.conv2 = ConvTranspose(128, 64, 5, 2, 2, 1)
        self.conv3 = ConvTranspose(64, self.output_channels, 5, 2, 2, 1)
        
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dense layer to expand latent dimension
        x = self.fc(x)
        x = self.relu(x)
        x = x.view(-1, 16, 4, 4, 4)  # Reshape to 4x4x4x16

        # Upsampling layers
        x = self.conv1(x)  # 4x4x4 -> 8x8x8
        x = self.relu(x)

        x = self.conv2(x)  # 8x8x8 -> 16x16x16
        x = self.relu(x)

        x = self.conv3(x)  # 16x16x16 -> 32x32x32
        x = _final_activation(x, self.color_mode, self.output_channels)
        
        # Reshape to match expected format (batch, depth, height, width, channels)
        x = x.permute(0, 2, 3, 4, 1)
        
        return x


class ProgressiveGenerator(nn.Module):
    """Progressive growing generator for high-resolution 3D data."""
    
    def __init__(self, void_dim: int = 64, noise_dim: int = 100, color_mode: int = 1, 
                 max_resolution: int = 128, sparse: bool = False):
        super().__init__()
        self.void_dim = void_dim
        self.noise_dim = noise_dim
        self.color_mode = color_mode
        self.max_resolution = max_resolution
        self.sparse = sparse
        
        self.output_channels = 6 if color_mode == 1 else 3
        
        # Progressive layers for different resolutions
        self.initial_block = self._make_initial_block()
        self.progressive_blocks = nn.ModuleList()
        
        # Create progressive blocks for each resolution level
        current_res = 8
        current_channels = noise_dim
        
        while current_res <= max_resolution:
            block = self._make_progressive_block(current_channels, current_channels // 2)
            self.progressive_blocks.append(block)
            current_channels = current_channels // 2
            current_res *= 2
        
        # Final output layer
        ConvTranspose = SparseConvTranspose3d if sparse else nn.ConvTranspose3d
        self.to_rgb = ConvTranspose(current_channels, self.output_channels, 1, 1, 0)
        
        self.current_level = 0  # Current progressive level
        self.alpha = 1.0  # Blending factor for progressive growing
    
    def _make_initial_block(self):
        """Create the initial 4x4x4 block."""
        ConvTranspose = SparseConvTranspose3d if self.sparse else nn.ConvTranspose3d
        BatchNorm = SparseBatchNorm3d if self.sparse else nn.BatchNorm3d
        
        return nn.Sequential(
            nn.Linear(self.noise_dim, 4 * 4 * 4 * self.noise_dim),
            nn.Unflatten(1, (self.noise_dim, 4, 4, 4)),
            ConvTranspose(self.noise_dim, self.noise_dim, 3, 1, 1),
            BatchNorm(self.noise_dim),
            nn.LeakyReLU(0.2)
        )
    
    def _make_progressive_block(self, in_channels: int, out_channels: int):
        """Create a progressive block that doubles the resolution."""
        ConvTranspose = SparseConvTranspose3d if self.sparse else nn.ConvTranspose3d
        BatchNorm = SparseBatchNorm3d if self.sparse else nn.BatchNorm3d
        
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            ConvTranspose(in_channels, out_channels, 3, 1, 1),
            BatchNorm(out_channels),
            nn.LeakyReLU(0.2),
            ConvTranspose(out_channels, out_channels, 3, 1, 1),
            BatchNorm(out_channels),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial block
        x = self.initial_block(x)
        
        # Progressive blocks up to current level
        for i in range(min(self.current_level + 1, len(self.progressive_blocks))):
            x = self.progressive_blocks[i](x)
        
        # Convert to output channels
        x = self.to_rgb(x)
        x = _final_activation(x, self.color_mode, self.output_channels)

        return x

    def grow(self):
        """Grow the network by one level."""
        if self.current_level < len(self.progressive_blocks) - 1:
            self.current_level += 1
            self.alpha = 0.0  # Start with full blend to new layer

    def set_alpha(self, alpha: float):
        """Set the blending factor for progressive growing."""
        self.alpha = max(0.0, min(1.0, alpha))


class ConditionalGenerator(nn.Module):
    """Conditional generator for controlled generation."""
    
    def __init__(self, void_dim: int = 64, noise_dim: int = 100, condition_dim: int = 10,
                 color_mode: int = 1, sparse: bool = False):
        super().__init__()
        self.void_dim = void_dim
        self.noise_dim = noise_dim
        self.condition_dim = condition_dim
        self.color_mode = color_mode
        self.sparse = sparse
        
        self.output_channels = 6 if color_mode == 1 else 3
        
        # Condition embedding
        self.condition_embedding = nn.Embedding(condition_dim, noise_dim)


# ============================================================================
# 3D U-Net Architecture for Diffusion Models
# ============================================================================

class SinusoidalPositionEmbedding(nn.Module):
    """
    Sinusoidal position embedding for time steps in diffusion models.
    
    This creates embeddings for timesteps that help the model understand
    the current noise level in the diffusion process.
    """
    
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal embeddings for timesteps.
        
        Args:
            timesteps: Tensor of shape (batch_size,) containing timestep values
            
        Returns:
            Embeddings of shape (batch_size, dim)
        """
        device = timesteps.device
        half_dim = self.dim // 2
        
        # Create frequency embeddings
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half_dim, device=device) / half_dim
        )
        
        # Apply to timesteps
        args = timesteps[:, None] * freqs[None, :]
        embeddings = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        # Handle odd dimensions
        if self.dim % 2 == 1:
            embeddings = torch.cat([embeddings, torch.zeros_like(embeddings[:, :1])], dim=-1)
            
        return embeddings


class TimeEmbedding(nn.Module):
    """
    Time embedding module that processes timestep embeddings for diffusion.
    
    Converts sinusoidal position embeddings into feature representations
    that can be injected into the U-Net at multiple scales.
    """
    
    def __init__(self, time_dim: int, hidden_dim: int):
        super().__init__()
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim
        
        self.time_embedding = SinusoidalPositionEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Process timesteps into embeddings.
        
        Args:
            timesteps: Tensor of shape (batch_size,)
            
        Returns:
            Time embeddings of shape (batch_size, hidden_dim)
        """
        time_emb = self.time_embedding(timesteps)
        time_emb = self.time_mlp(time_emb)
        return time_emb


class ResidualBlock3D(nn.Module):
    """
    3D Residual block with time embedding injection and optional sparse tensor support.
    
    This block forms the core building component of the U-Net, providing
    stable training through residual connections and time conditioning.
    """
    
    def __init__(self, in_channels: int, out_channels: int, time_dim: int,
                 kernel_size: int = 3, padding: int = 1, sparse: bool = False,
                 dropout: float = 0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sparse = sparse
        
        # Choose layer types based on sparse mode
        Conv3d = SparseConv3d if sparse else nn.Conv3d
        BatchNorm = SparseBatchNorm3d if sparse else nn.BatchNorm3d
        
        # First convolution block
        self.conv1 = Conv3d(in_channels, out_channels, kernel_size, padding=padding)
        self.norm1 = BatchNorm(out_channels)
        
        # Time embedding projection
        self.time_proj = nn.Linear(time_dim, out_channels)
        
        # Second convolution block
        self.conv2 = Conv3d(out_channels, out_channels, kernel_size, padding=padding)
        self.norm2 = BatchNorm(out_channels)
        
        # Residual connection
        if in_channels != out_channels:
            self.residual_conv = Conv3d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()
            
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with time embedding injection.
        
        Args:
            x: Input tensor of shape (batch, channels, depth, height, width)
            time_emb: Time embedding of shape (batch, time_dim)
            
        Returns:
            Output tensor with same spatial dimensions as input
        """
        residual = self.residual_conv(x)
        
        # First conv block
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        
        # Inject time embedding
        time_proj = self.time_proj(time_emb)
        # Reshape time embedding to match spatial dimensions
        time_proj = time_proj.view(time_proj.shape[0], time_proj.shape[1], 1, 1, 1)
        out = out + time_proj
        
        # Second conv block
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.dropout(out)
        
        # Residual connection
        out = out + residual
        out = self.activation(out)
        
        return out


class AttentionBlock3D(nn.Module):
    """
    3D Self-attention block for improved feature representation in U-Net.
    
    Implements multi-head self-attention adapted for 3D data, helping the model
    capture long-range dependencies in the 3D structure.
    """
    
    def __init__(self, channels: int, num_heads: int = 8, sparse: bool = False):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.sparse = sparse
        
        assert channels % num_heads == 0, "Channels must be divisible by num_heads"
        self.head_dim = channels // num_heads
        
        # Choose normalization based on sparse mode
        BatchNorm = SparseBatchNorm3d if sparse else nn.BatchNorm3d
        self.norm = BatchNorm(channels)
        
        # Attention projections
        self.to_qkv = nn.Conv3d(channels, channels * 3, 1)
        self.to_out = nn.Conv3d(channels, channels, 1)
        
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply 3D self-attention.
        
        Args:
            x: Input tensor of shape (batch, channels, depth, height, width)
            
        Returns:
            Attention-processed tensor of same shape
        """
        batch, channels, depth, height, width = x.shape
        residual = x
        
        # Normalize input
        x = self.norm(x)
        
        # Generate queries, keys, values
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for multi-head attention
        q = q.view(batch, self.num_heads, self.head_dim, depth * height * width)
        k = k.view(batch, self.num_heads, self.head_dim, depth * height * width)
        v = v.view(batch, self.num_heads, self.head_dim, depth * height * width)
        
        # Compute attention
        q = q.transpose(-2, -1)  # (batch, heads, spatial, head_dim)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        
        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(-2, -1)  # (batch, heads, head_dim, spatial)
        out = out.contiguous().view(batch, channels, depth, height, width)
        
        # Output projection
        out = self.to_out(out)
        
        # Residual connection
        return out + residual


class UNet3DEncoder(nn.Module):
    """
    3D U-Net encoder with multi-scale feature extraction.
    
    The encoder progressively downsamples the input while increasing
    the number of channels, capturing features at multiple scales.
    """
    
    def __init__(self, in_channels: int, base_channels: int = 64, 
                 time_dim: int = 256, num_levels: int = 4,
                 sparse: bool = False, use_attention: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.time_dim = time_dim
        self.num_levels = num_levels
        self.sparse = sparse
        self.use_attention = use_attention
        
        # Initial convolution
        Conv3d = SparseConv3d if sparse else nn.Conv3d
        self.initial_conv = Conv3d(in_channels, base_channels, 3, padding=1)
        
        # Encoder levels
        self.encoder_blocks = nn.ModuleList()
        self.downsample_blocks = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()
        
        current_channels = base_channels
        
        for level in range(num_levels):
            # Residual blocks for this level
            blocks = nn.ModuleList([
                ResidualBlock3D(current_channels, current_channels, time_dim, sparse=sparse),
                ResidualBlock3D(current_channels, current_channels, time_dim, sparse=sparse)
            ])
            self.encoder_blocks.append(blocks)
            
            # Attention block (only at certain levels to manage computation)
            if use_attention and level >= num_levels // 2:
                self.attention_blocks.append(AttentionBlock3D(current_channels, sparse=sparse))
            else:
                self.attention_blocks.append(nn.Identity())
            
            # Downsampling (except for the last level)
            if level < num_levels - 1:
                next_channels = current_channels * 2
                downsample = Conv3d(current_channels, next_channels, 3, stride=2, padding=1)
                self.downsample_blocks.append(downsample)
                current_channels = next_channels
            else:
                self.downsample_blocks.append(nn.Identity())
                
        self.final_channels = current_channels
        
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Encode input through multiple scales.
        
        Args:
            x: Input tensor of shape (batch, channels, depth, height, width)
            time_emb: Time embedding of shape (batch, time_dim)
            
        Returns:
            Tuple of (final_features, skip_connections)
        """
        # Initial convolution
        x = self.initial_conv(x)
        
        skip_connections = []
        
        # Process through encoder levels
        for level in range(self.num_levels):
            # Apply residual blocks
            for block in self.encoder_blocks[level]:
                x = block(x, time_emb)
            
            # Apply attention if present
            x = self.attention_blocks[level](x)
            
            # Store skip connection
            skip_connections.append(x)
            
            # Downsample (except for last level)
            if level < self.num_levels - 1:
                x = self.downsample_blocks[level](x)
                
        return x, skip_connections


class UNet3DDecoder(nn.Module):
    """
    3D U-Net decoder with skip connections and upsampling.
    
    The decoder progressively upsamples features while incorporating
    skip connections from the encoder for detailed reconstruction.
    """
    
    def __init__(self, encoder_channels: List[int], out_channels: int,
                 time_dim: int = 256, sparse: bool = False, use_attention: bool = True):
        super().__init__()
        self.encoder_channels = encoder_channels
        self.out_channels = out_channels
        self.time_dim = time_dim
        self.sparse = sparse
        self.use_attention = use_attention
        
        # Choose layer types based on sparse mode
        Conv3d = SparseConv3d if sparse else nn.Conv3d
        ConvTranspose3d = SparseConvTranspose3d if sparse else nn.ConvTranspose3d
        
        # Decoder levels (reverse order of encoder)
        self.decoder_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()
        self.skip_conv_blocks = nn.ModuleList()
        
        # Reverse the channel list for decoder
        decoder_channels = encoder_channels[::-1]
        
        for level in range(len(decoder_channels) - 1):
            current_channels = decoder_channels[level]
            next_channels = decoder_channels[level + 1]
            skip_channels = next_channels  # Skip connection from encoder
            
            # Skip connection processing
            skip_conv = Conv3d(skip_channels, next_channels, 1)
            self.skip_conv_blocks.append(skip_conv)
            
            # Upsampling
            upsample = ConvTranspose3d(current_channels, next_channels, 3, stride=2, padding=1, output_padding=1)
            self.upsample_blocks.append(upsample)
            
            # Residual blocks after concatenation
            concat_channels = next_channels * 2  # Upsampled + skip connection
            blocks = nn.ModuleList([
                ResidualBlock3D(concat_channels, next_channels, time_dim, sparse=sparse),
                ResidualBlock3D(next_channels, next_channels, time_dim, sparse=sparse)
            ])
            self.decoder_blocks.append(blocks)
            
            # Attention blocks (at certain levels)
            if use_attention and level < len(decoder_channels) // 2:
                self.attention_blocks.append(AttentionBlock3D(next_channels, sparse=sparse))
            else:
                self.attention_blocks.append(nn.Identity())
        
        # Final output convolution
        final_channels = decoder_channels[-1]
        self.final_conv = Conv3d(final_channels, out_channels, 3, padding=1)
        
    def forward(self, x: torch.Tensor, skip_connections: List[torch.Tensor], 
                time_emb: torch.Tensor) -> torch.Tensor:
        """
        Decode features with skip connections.
        
        Args:
            x: Encoded features from encoder
            skip_connections: List of skip connection tensors from encoder
            time_emb: Time embedding of shape (batch, time_dim)
            
        Returns:
            Decoded output tensor
        """
        # Reverse skip connections to match decoder order
        skip_connections = skip_connections[::-1][1:]  # Exclude the deepest level
        
        for level in range(len(self.decoder_blocks)):
            # Upsample current features
            x = self.upsample_blocks[level](x)
            
            # Process skip connection
            skip = skip_connections[level]
            skip = self.skip_conv_blocks[level](skip)
            
            # Ensure spatial dimensions match before concatenation
            if x.shape[2:] != skip.shape[2:]:
                # Resize skip connection to match upsampled features
                skip = F.interpolate(skip, size=x.shape[2:], mode='trilinear', align_corners=False)
            
            # Concatenate upsampled features with skip connection
            x = torch.cat([x, skip], dim=1)
            
            # Apply residual blocks
            for block in self.decoder_blocks[level]:
                x = block(x, time_emb)
            
            # Apply attention if present
            x = self.attention_blocks[level](x)
        
        # Final output convolution
        x = self.final_conv(x)
        
        return x


class UNet3D(nn.Module):
    """
    Complete 3D U-Net architecture for diffusion models.
    
    This model combines the encoder and decoder with time embedding
    and optional conditioning for controlled 3D generation.
    """
    
    def __init__(self, in_channels: int = 4, out_channels: int = 4,
                 base_channels: int = 64, time_dim: int = 256,
                 num_levels: int = 4, sparse: bool = False,
                 use_attention: bool = True, condition_dim: Optional[int] = None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.time_dim = time_dim
        self.num_levels = num_levels
        self.sparse = sparse
        self.use_attention = use_attention
        self.condition_dim = condition_dim
        
        # Time embedding
        self.time_embedding = TimeEmbedding(time_dim, time_dim)
        
        # Conditioning embedding (optional)
        if condition_dim is not None:
            self.condition_embedding = nn.Sequential(
                nn.Linear(condition_dim, time_dim),
                nn.SiLU(),
                nn.Linear(time_dim, time_dim)
            )
        else:
            self.condition_embedding = None
        
        # Calculate encoder channel progression
        encoder_channels = [base_channels * (2 ** i) for i in range(num_levels)]
        encoder_channels[0] = base_channels  # First level uses base channels
        
        # Encoder
        self.encoder = UNet3DEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            time_dim=time_dim,
            num_levels=num_levels,
            sparse=sparse,
            use_attention=use_attention
        )
        
        # Decoder
        self.decoder = UNet3DDecoder(
            encoder_channels=encoder_channels,
            out_channels=out_channels,
            time_dim=time_dim,
            sparse=sparse,
            use_attention=use_attention
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d, nn.Linear)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm3d, nn.GroupNorm)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor,
                condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the 3D U-Net.
        
        Args:
            x: Input tensor of shape (batch, channels, depth, height, width)
            timesteps: Timestep tensor of shape (batch,)
            condition: Optional conditioning tensor of shape (batch, condition_dim)
            
        Returns:
            Output tensor of same spatial shape as input
        """
        # Process time embedding
        time_emb = self.time_embedding(timesteps)
        
        # Add conditioning if provided
        if condition is not None and self.condition_embedding is not None:
            condition_emb = self.condition_embedding(condition)
            time_emb = time_emb + condition_emb
        
        # Encode
        encoded_features, skip_connections = self.encoder(x, time_emb)
        
        # Decode
        output = self.decoder(encoded_features, skip_connections, time_emb)
        
        return output
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics for the model."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Estimate memory usage (rough approximation)
        param_memory = total_params * 4  # 4 bytes per float32 parameter
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "estimated_param_memory_mb": param_memory / (1024 * 1024),
            "sparse_mode": self.sparse,
            "attention_enabled": self.use_attention,
            "num_levels": self.num_levels,
            "base_channels": self.base_channels
        }
    
    def optimize_for_resolution(self, target_resolution: int):
        """
        Optimize model architecture for target resolution.
        
        Args:
            target_resolution: Target 3D resolution (e.g., 64 for 64x64x64)
        """
        # Calculate optimal number of levels based on resolution
        optimal_levels = int(math.log2(target_resolution)) - 2  # Leave room for base resolution
        optimal_levels = max(3, min(6, optimal_levels))  # Clamp between 3 and 6
        
        if optimal_levels != self.num_levels:
            print(f"Recommended num_levels for resolution {target_resolution}: {optimal_levels}")
            print(f"Current num_levels: {self.num_levels}")
        
        # Calculate optimal base channels
        # Higher resolution needs more channels for detail capture
        if target_resolution >= 128:
            recommended_base = 32
        elif target_resolution >= 64:
            recommended_base = 64
        else:
            recommended_base = 128
            
        if recommended_base != self.base_channels:
            print(f"Recommended base_channels for resolution {target_resolution}: {recommended_base}")
            print(f"Current base_channels: {self.base_channels}")
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing to save memory during training."""
        def checkpoint_forward(module):
            if hasattr(module, 'forward'):
                original_forward = module.forward
                def checkpointed_forward(*args, **kwargs):
                    return torch.utils.checkpoint.checkpoint(original_forward, *args, **kwargs)
                module.forward = checkpointed_forward
        
        # Apply to encoder and decoder blocks
        for block in self.encoder.encoder_blocks:
            for subblock in block:
                checkpoint_forward(subblock)
        
        for block in self.decoder.decoder_blocks:
            for subblock in block:
                checkpoint_forward(subblock)


# ============================================================================
# Advanced Conditioning Mechanisms
# ============================================================================

class CrossAttentionBlock3D(nn.Module):
    """
    3D Cross-attention block for conditioning on external features.
    
    This allows the model to attend to conditioning information
    such as text embeddings, parameter vectors, or other modalities.
    """
    
    def __init__(self, channels: int, condition_dim: int, num_heads: int = 8, sparse: bool = False):
        super().__init__()
        self.channels = channels
        self.condition_dim = condition_dim
        self.num_heads = num_heads
        self.sparse = sparse
        
        assert channels % num_heads == 0, "Channels must be divisible by num_heads"
        self.head_dim = channels // num_heads
        
        # Choose normalization based on sparse mode
        BatchNorm = SparseBatchNorm3d if sparse else nn.BatchNorm3d
        self.norm_x = BatchNorm(channels)
        self.norm_condition = nn.LayerNorm(condition_dim)
        
        # Attention projections
        self.to_q = nn.Conv3d(channels, channels, 1)
        self.to_k = nn.Linear(condition_dim, channels)
        self.to_v = nn.Linear(condition_dim, channels)
        self.to_out = nn.Conv3d(channels, channels, 1)
        
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Apply cross-attention between 3D features and conditioning.
        
        Args:
            x: Input tensor of shape (batch, channels, depth, height, width)
            condition: Conditioning tensor of shape (batch, seq_len, condition_dim)
            
        Returns:
            Cross-attention processed tensor of same shape as x
        """
        batch, channels, depth, height, width = x.shape
        residual = x
        
        # Normalize inputs
        x = self.norm_x(x)
        condition = self.norm_condition(condition)
        
        # Generate queries from spatial features
        q = self.to_q(x)
        q = q.view(batch, self.num_heads, self.head_dim, depth * height * width)
        q = q.transpose(-2, -1)  # (batch, heads, spatial, head_dim)
        
        # Generate keys and values from conditioning
        k = self.to_k(condition)  # (batch, seq_len, channels)
        v = self.to_v(condition)
        
        k = k.view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, heads, seq_len, head_dim)
        v = v.view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute cross-attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)  # (batch, heads, spatial, head_dim)
        out = out.transpose(-2, -1).contiguous()  # (batch, heads, head_dim, spatial)
        out = out.view(batch, channels, depth, height, width)
        
        # Output projection
        out = self.to_out(out)
        
        # Residual connection
        return out + residual


class MultiModalConditioner(nn.Module):
    """
    Multi-modal conditioning system supporting various input types.
    
    This module can handle different types of conditioning inputs:
    - Text embeddings
    - Parameter vectors (shape, material properties, etc.)
    - Image conditioning
    - Categorical labels
    """
    
    def __init__(self, time_dim: int = 256, text_dim: int = 512, 
                 param_dim: int = 64, image_channels: int = 3,
                 num_categories: int = 10):
        super().__init__()
        self.time_dim = time_dim
        self.text_dim = text_dim
        self.param_dim = param_dim
        self.image_channels = image_channels
        self.num_categories = num_categories
        
        # Text conditioning
        self.text_processor = nn.Sequential(
            nn.Linear(text_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Parameter conditioning
        self.param_processor = nn.Sequential(
            nn.Linear(param_dim, time_dim // 2),
            nn.SiLU(),
            nn.Linear(time_dim // 2, time_dim)
        )
        
        # Image conditioning (for 2D reference images)
        self.image_processor = nn.Sequential(
            nn.Conv2d(image_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, time_dim)
        )
        
        # Category conditioning
        self.category_embedding = nn.Embedding(num_categories, time_dim)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(time_dim * 4, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim)
        )
        
    def forward(self, text_emb: Optional[torch.Tensor] = None,
                params: Optional[torch.Tensor] = None,
                image: Optional[torch.Tensor] = None,
                category: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Process multiple conditioning modalities.
        
        Args:
            text_emb: Text embeddings of shape (batch, text_dim)
            params: Parameter vector of shape (batch, param_dim)
            image: Reference image of shape (batch, channels, height, width)
            category: Category indices of shape (batch,)
            
        Returns:
            Fused conditioning embedding of shape (batch, time_dim)
        """
        batch_size = None
        conditions = []
        
        # Process text conditioning
        if text_emb is not None:
            batch_size = text_emb.shape[0]
            text_cond = self.text_processor(text_emb)
            conditions.append(text_cond)
        else:
            if batch_size is None:
                raise ValueError("At least one conditioning input must be provided")
            conditions.append(torch.zeros(batch_size, self.time_dim, device=text_emb.device if text_emb is not None else 'cpu'))
        
        # Process parameter conditioning
        if params is not None:
            if batch_size is None:
                batch_size = params.shape[0]
            param_cond = self.param_processor(params)
            conditions.append(param_cond)
        else:
            device = conditions[0].device if conditions else 'cpu'
            conditions.append(torch.zeros(batch_size, self.time_dim, device=device))
        
        # Process image conditioning
        if image is not None:
            if batch_size is None:
                batch_size = image.shape[0]
            image_cond = self.image_processor(image)
            conditions.append(image_cond)
        else:
            device = conditions[0].device if conditions else 'cpu'
            conditions.append(torch.zeros(batch_size, self.time_dim, device=device))
        
        # Process category conditioning
        if category is not None:
            if batch_size is None:
                batch_size = category.shape[0]
            cat_cond = self.category_embedding(category)
            conditions.append(cat_cond)
        else:
            device = conditions[0].device if conditions else 'cpu'
            conditions.append(torch.zeros(batch_size, self.time_dim, device=device))
        
        # Fuse all conditions
        fused_condition = torch.cat(conditions, dim=-1)
        fused_condition = self.fusion(fused_condition)
        
        return fused_condition


class ConditionalUNet3D(UNet3D):
    """
    Enhanced 3D U-Net with advanced conditioning capabilities.
    
    This extends the base U-Net with cross-attention mechanisms
    and multi-modal conditioning support.
    """
    
    def __init__(self, in_channels: int = 4, out_channels: int = 4,
                 base_channels: int = 64, time_dim: int = 256,
                 num_levels: int = 4, sparse: bool = False,
                 use_attention: bool = True, use_cross_attention: bool = True,
                 text_dim: int = 512, param_dim: int = 64,
                 image_channels: int = 3, num_categories: int = 10):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            time_dim=time_dim,
            num_levels=num_levels,
            sparse=sparse,
            use_attention=use_attention,
            condition_dim=None  # We'll handle conditioning differently
        )
        
        self.use_cross_attention = use_cross_attention
        
        # Multi-modal conditioner
        self.multi_modal_conditioner = MultiModalConditioner(
            time_dim=time_dim,
            text_dim=text_dim,
            param_dim=param_dim,
            image_channels=image_channels,
            num_categories=num_categories
        )
        
        # Cross-attention blocks (add to encoder and decoder)
        if use_cross_attention:
            self.encoder_cross_attn = nn.ModuleList()
            self.decoder_cross_attn = nn.ModuleList()
            
            # Add cross-attention to encoder
            encoder_channels = [base_channels * (2 ** i) for i in range(num_levels)]
            for channels in encoder_channels:
                self.encoder_cross_attn.append(
                    CrossAttentionBlock3D(channels, time_dim, sparse=sparse)
                )
            
            # Add cross-attention to decoder
            decoder_channels = encoder_channels[::-1][1:]  # Reverse and exclude deepest
            for channels in decoder_channels:
                self.decoder_cross_attn.append(
                    CrossAttentionBlock3D(channels, time_dim, sparse=sparse)
                )
        
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor,
                text_emb: Optional[torch.Tensor] = None,
                params: Optional[torch.Tensor] = None,
                image: Optional[torch.Tensor] = None,
                category: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with multi-modal conditioning.
        
        Args:
            x: Input tensor of shape (batch, channels, depth, height, width)
            timesteps: Timestep tensor of shape (batch,)
            text_emb: Text embeddings of shape (batch, text_dim)
            params: Parameter vector of shape (batch, param_dim)
            image: Reference image of shape (batch, channels, height, width)
            category: Category indices of shape (batch,)
            
        Returns:
            Output tensor of same spatial shape as input
        """
        # Process time embedding
        time_emb = self.time_embedding(timesteps)
        
        # Process multi-modal conditioning
        if any(cond is not None for cond in [text_emb, params, image, category]):
            condition_emb = self.multi_modal_conditioner(text_emb, params, image, category)
            time_emb = time_emb + condition_emb
            
            # Prepare conditioning for cross-attention
            # Expand conditioning to sequence format for cross-attention
            condition_seq = condition_emb.unsqueeze(1)  # (batch, 1, time_dim)
        else:
            condition_seq = None
        
        # Encode with cross-attention
        x = self.encoder.initial_conv(x)
        skip_connections = []
        
        for level in range(self.encoder.num_levels):
            # Apply residual blocks
            for block in self.encoder.encoder_blocks[level]:
                x = block(x, time_emb)
            
            # Apply self-attention
            x = self.encoder.attention_blocks[level](x)
            
            # Apply cross-attention if enabled and conditioning is available
            if self.use_cross_attention and condition_seq is not None and level < len(self.encoder_cross_attn):
                x = self.encoder_cross_attn[level](x, condition_seq)
            
            # Store skip connection
            skip_connections.append(x)
            
            # Downsample (except for last level)
            if level < self.encoder.num_levels - 1:
                x = self.encoder.downsample_blocks[level](x)
        
        # Decode with cross-attention
        skip_connections = skip_connections[::-1][1:]  # Reverse and exclude deepest
        
        for level in range(len(self.decoder.decoder_blocks)):
            # Upsample current features
            x = self.decoder.upsample_blocks[level](x)
            
            # Process skip connection
            skip = skip_connections[level]
            skip = self.decoder.skip_conv_blocks[level](skip)
            
            # Concatenate upsampled features with skip connection
            x = torch.cat([x, skip], dim=1)
            
            # Apply residual blocks
            for block in self.decoder.decoder_blocks[level]:
                x = block(x, time_emb)
            
            # Apply self-attention
            x = self.decoder.attention_blocks[level](x)
            
            # Apply cross-attention if enabled and conditioning is available
            if self.use_cross_attention and condition_seq is not None and level < len(self.decoder_cross_attn):
                x = self.decoder_cross_attn[level](x, condition_seq)
        
        # Final output convolution
        x = self.decoder.final_conv(x)
        
        return x

# ============================================================================
# DISCRIMINATOR MODELS
# ============================================================================

class SimpleDiscriminator(nn.Module):
    """Simple discriminator model equivalent to TensorFlow version."""
    
    def __init__(self, void_dim: int = 64, noise_dim: int = 100, color_mode: int = 1, sparse: bool = False):
        super().__init__()
        self.void_dim = void_dim
        self.noise_dim = noise_dim
        self.color_mode = color_mode
        self.sparse = sparse
        
        # Input channels based on color mode
        self.input_channels = 6 if color_mode == 1 else 3
        
        # Convolution layers
        Conv = SparseConv3d if sparse else nn.Conv3d
        
        self.conv1 = Conv(self.input_channels, noise_dim // 8, 3, 2, 1)
        self.conv2 = Conv(noise_dim // 8, noise_dim // 4, 3, 2, 1)
        self.conv3 = Conv(noise_dim // 4, noise_dim // 2, 3, 2, 1)
        self.conv4 = Conv(noise_dim // 2, noise_dim, 3, 2, 1)
        
        # Calculate the size after convolutions
        final_size = void_dim // (2 ** 4)  # 4 conv layers with stride 2
        self.fc = nn.Linear(noise_dim * final_size ** 3, 1)
        
        self.leaky_relu = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape input from (batch, void_dim, void_dim, void_dim, channels) to (batch, channels, void_dim, void_dim, void_dim)
        x = x.permute(0, 4, 1, 2, 3)
        
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        # Flatten and final linear layer — return logits for softplus loss
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x


class ComplexDiscriminator(nn.Module):
    """Complex discriminator model with enhanced architecture."""
    
    def __init__(self, void_dim: int = 64, noise_dim: int = 100, color_mode: int = 1, sparse: bool = False):
        super().__init__()
        self.void_dim = void_dim
        self.noise_dim = noise_dim
        self.color_mode = color_mode
        self.sparse = sparse
        
        self.input_channels = 6 if color_mode == 1 else 3
        
        # Enhanced architecture with more layers and skip connections
        Conv = SparseConv3d if sparse else nn.Conv3d
        BatchNorm = SparseBatchNorm3d if sparse else nn.BatchNorm3d
        
        # First block
        self.conv1 = Conv(self.input_channels, noise_dim // 8, 3, 2, 1)
        self.bn1 = BatchNorm(noise_dim // 8)
        
        # Second block
        self.conv2 = Conv(noise_dim // 8, noise_dim // 4, 3, 2, 1)
        self.bn2 = BatchNorm(noise_dim // 4)
        
        # Third block with attention
        self.conv3 = Conv(noise_dim // 4, noise_dim // 2, 3, 2, 1)
        self.bn3 = BatchNorm(noise_dim // 2)
        self.attention = SelfAttention3D(noise_dim // 2)
        
        # Fourth block
        self.conv4 = Conv(noise_dim // 2, noise_dim, 3, 2, 1)
        self.bn4 = BatchNorm(noise_dim)
        
        # Fifth block for higher capacity
        self.conv5 = Conv(noise_dim, noise_dim * 2, 3, 2, 1)
        self.bn5 = BatchNorm(noise_dim * 2)
        
        # Calculate final size
        final_size = void_dim // (2 ** 5)
        self.fc1 = nn.Linear(noise_dim * 2 * final_size ** 3, 512)
        self.fc2 = nn.Linear(512, 1)
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape input
        x = x.permute(0, 4, 1, 2, 3)
        
        # First block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        # Second block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        # Third block with attention
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)
        x = self.attention(x)
        x = self.dropout(x)
        
        # Fourth block
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        # Fifth block
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        # Flatten and final layers
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        # Return logits for softplus loss
        x = self.fc2(x)

        return x


class SkipDiscriminator(nn.Module):
    """Discriminator for the skip connection model."""
    
    def __init__(self, void_dim: int = 64, noise_dim: int = 100, color_mode: int = 1, sparse: bool = False):
        super().__init__()
        # Use the same architecture as SimpleDiscriminator for now
        self.discriminator = SimpleDiscriminator(void_dim, noise_dim, color_mode, sparse)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.discriminator(x)


class MonochromeDiscriminator(nn.Module):
    """Discriminator for monochrome models."""
    
    def __init__(self, void_dim: int = 64, noise_dim: int = 100, color_mode: int = 0, sparse: bool = False):
        super().__init__()
        # Use SimpleDiscriminator with color_mode=0 (3 channels)
        self.discriminator = SimpleDiscriminator(void_dim, noise_dim, 0, sparse)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.discriminator(x)


class AutoencoderDiscriminator(nn.Module):
    """Discriminator for autoencoder architecture."""
    
    def __init__(self, void_dim: int = 64, noise_dim: int = 100, color_mode: int = 1, sparse: bool = False):
        super().__init__()
        self.void_dim = void_dim
        self.noise_dim = noise_dim
        self.color_mode = color_mode
        
        # Input channels based on color mode
        self.input_channels = 6 if color_mode == 1 else 3
        
        # Use a simple CNN-based discriminator for 3D data
        Conv = SparseConv3d if sparse else nn.Conv3d
        
        self.conv1 = Conv(self.input_channels, 32, 4, 2, 1)
        self.conv2 = Conv(32, 64, 4, 2, 1)
        self.conv3 = Conv(64, 128, 4, 2, 1)
        
        # Calculate the size after convolutions for the autoencoder output (32x32x32)
        # After 3 conv layers with stride 2: 32 -> 16 -> 8 -> 4
        final_size = 4
        self.fc = nn.Linear(128 * final_size ** 3, 1)
        
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape input from (batch, depth, height, width, channels) to (batch, channels, depth, height, width)
        x = x.permute(0, 4, 1, 2, 3)

        x = self.conv1(x)
        x = self.leaky_relu(x)

        x = self.conv2(x)
        x = self.leaky_relu(x)

        x = self.conv3(x)
        x = self.leaky_relu(x)

        # Flatten and final linear layer — return logits for softplus loss
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x


class SpectralNormDiscriminator(nn.Module):
    """Discriminator with spectral normalization for training stability."""
    
    def __init__(self, void_dim: int = 64, noise_dim: int = 100, color_mode: int = 1, sparse: bool = False):
        super().__init__()
        self.void_dim = void_dim
        self.noise_dim = noise_dim
        self.color_mode = color_mode
        self.sparse = sparse
        
        self.input_channels = 6 if color_mode == 1 else 3
        
        # Convolution layers with spectral normalization
        Conv = SparseConv3d if sparse else nn.Conv3d
        
        self.conv1 = nn.utils.spectral_norm(Conv(self.input_channels, noise_dim // 8, 3, 2, 1))
        self.conv2 = nn.utils.spectral_norm(Conv(noise_dim // 8, noise_dim // 4, 3, 2, 1))
        self.conv3 = nn.utils.spectral_norm(Conv(noise_dim // 4, noise_dim // 2, 3, 2, 1))
        self.conv4 = nn.utils.spectral_norm(Conv(noise_dim // 2, noise_dim, 3, 2, 1))
        
        final_size = void_dim // (2 ** 4)
        self.fc = nn.utils.spectral_norm(nn.Linear(noise_dim * final_size ** 3, 1))
        
        self.leaky_relu = nn.LeakyReLU(0.2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape input
        x = x.permute(0, 4, 1, 2, 3)
        
        x = self.conv1(x)
        x = self.leaky_relu(x)
        
        x = self.conv2(x)
        x = self.leaky_relu(x)
        
        x = self.conv3(x)
        x = self.leaky_relu(x)
        
        x = self.conv4(x)
        x = self.leaky_relu(x)
        
        # Flatten and final linear layer
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        
        return x


class ProgressiveDiscriminator(nn.Module):
    """Progressive discriminator for high-resolution 3D data."""
    
    def __init__(self, void_dim: int = 64, noise_dim: int = 100, color_mode: int = 1,
                 max_resolution: int = 128, sparse: bool = False):
        super().__init__()
        self.void_dim = void_dim
        self.noise_dim = noise_dim
        self.color_mode = color_mode
        self.max_resolution = max_resolution
        self.sparse = sparse
        
        self.input_channels = 6 if color_mode == 1 else 3
        
        # Progressive blocks for different resolutions
        self.progressive_blocks = nn.ModuleList()
        self.from_rgb_layers = nn.ModuleList()
        
        # Create progressive blocks for each resolution level
        current_res = max_resolution
        current_channels = noise_dim // 8
        
        while current_res >= 8:
            # From RGB layer for this resolution
            Conv = SparseConv3d if sparse else nn.Conv3d
            from_rgb = Conv(self.input_channels, current_channels, 1, 1, 0)
            self.from_rgb_layers.append(from_rgb)
            
            # Progressive block
            block = self._make_progressive_block(current_channels, current_channels * 2)
            self.progressive_blocks.append(block)
            
            current_channels *= 2
            current_res //= 2
        
        # Final block
        self.final_block = self._make_final_block(current_channels)
        
        self.current_level = 0  # Current progressive level
        self.alpha = 1.0  # Blending factor
    
    def _make_progressive_block(self, in_channels: int, out_channels: int):
        """Create a progressive block that halves the resolution."""
        Conv = SparseConv3d if self.sparse else nn.Conv3d
        
        return nn.Sequential(
            Conv(in_channels, in_channels, 3, 1, 1),
            nn.LeakyReLU(0.2),
            Conv(in_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.AvgPool3d(2)
        )
    
    def _make_final_block(self, in_channels: int):
        """Create the final classification block."""
        return nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape input
        x = x.permute(0, 4, 1, 2, 3)
        
        # Convert from RGB at current resolution
        x = self.from_rgb_layers[self.current_level](x)
        x = nn.LeakyReLU(0.2)(x)
        
        # Progressive blocks from current level
        for i in range(self.current_level, len(self.progressive_blocks)):
            x = self.progressive_blocks[i](x)
        
        # Final block
        x = self.final_block(x)
        
        return x
    
    def grow(self):
        """Grow the network by one level."""
        if self.current_level < len(self.progressive_blocks) - 1:
            self.current_level += 1
            self.alpha = 0.0
    
    def set_alpha(self, alpha: float):
        """Set the blending factor for progressive growing."""
        self.alpha = max(0.0, min(1.0, alpha))


class MultiScaleDiscriminator(nn.Module):
    """Multi-scale discriminator for feature matching."""
    
    def __init__(self, void_dim: int = 64, noise_dim: int = 100, color_mode: int = 1, 
                 num_scales: int = 3, sparse: bool = False):
        super().__init__()
        self.num_scales = num_scales
        
        # Create discriminators for different scales
        self.discriminators = nn.ModuleList()
        for i in range(num_scales):
            scale_dim = void_dim // (2 ** i)
            disc = SimpleDiscriminator(scale_dim, noise_dim, color_mode, sparse)
            self.discriminators.append(disc)
        
        self.downsample = nn.AvgPool3d(2, stride=2)
    
    def forward(self, x: torch.Tensor) -> list:
        outputs = []
        current_x = x
        
        for i, discriminator in enumerate(self.discriminators):
            if i > 0:
                # Downsample for smaller scales
                current_x = self.downsample(current_x.permute(0, 4, 1, 2, 3))
                current_x = current_x.permute(0, 2, 3, 4, 1)
            
            output = discriminator(current_x)
            outputs.append(output)
        
        return outputs


class SelfAttention3D(nn.Module):
    """Self-attention mechanism for 3D data."""
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        
        self.query_conv = nn.Conv3d(in_channels, in_channels // 8, 1)
        self.key_conv = nn.Conv3d(in_channels, in_channels // 8, 1)
        self.value_conv = nn.Conv3d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, depth, height, width = x.size()
        
        # Generate query, key, value
        proj_query = self.query_conv(x).view(batch_size, -1, depth * height * width).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, depth * height * width)
        proj_value = self.value_conv(x).view(batch_size, -1, depth * height * width)
        
        # Attention
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        
        # Apply attention
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, depth, height, width)
        
        # Residual connection
        out = self.gamma * out + x
        
        return out


class ConditionalDiscriminator(nn.Module):
    """Conditional discriminator for controlled generation."""
    
    def __init__(self, void_dim: int = 64, noise_dim: int = 100, condition_dim: int = 10,
                 color_mode: int = 1, sparse: bool = False):
        super().__init__()
        self.void_dim = void_dim
        self.noise_dim = noise_dim
        self.condition_dim = condition_dim
        self.color_mode = color_mode
        self.sparse = sparse
        
        self.input_channels = 6 if color_mode == 1 else 3
        
        # Condition embedding
        self.condition_embedding = nn.Embedding(condition_dim, void_dim ** 3)
        
        # Modified input channels to include condition
        Conv = SparseConv3d if sparse else nn.Conv3d
        
        self.conv1 = Conv(self.input_channels + 1, noise_dim // 8, 3, 2, 1)  # +1 for condition
        self.conv2 = Conv(noise_dim // 8, noise_dim // 4, 3, 2, 1)
        self.conv3 = Conv(noise_dim // 4, noise_dim // 2, 3, 2, 1)
        self.conv4 = Conv(noise_dim // 2, noise_dim, 3, 2, 1)
        
        final_size = void_dim // (2 ** 4)
        self.fc = nn.Linear(noise_dim * final_size ** 3, 1)
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        # Embed condition and reshape to match input dimensions
        condition_emb = self.condition_embedding(condition)
        condition_emb = condition_emb.view(-1, self.void_dim, self.void_dim, self.void_dim, 1)

        # Concatenate input and condition
        x = torch.cat([x, condition_emb], dim=-1)

        # Reshape for convolution
        x = x.permute(0, 4, 1, 2, 3)

        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        # Flatten and final linear layer — return logits for softplus loss
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x
# ============================================================================
# MODEL FACTORY AND UTILITIES
# ============================================================================

class PyTorchModelFactory:
    """Factory class for creating PyTorch 3D generation models with backward compatibility."""
    
    # Model registry for generators
    GENERATOR_REGISTRY = {
        "simple": SimpleGenerator,
        "complex": ComplexGenerator,
        "skip": SkipGenerator,
        "monochrome": MonochromeGenerator,
        "autoencoder": AutoencoderGenerator,
        "progressive": ProgressiveGenerator,
        "conditional": ConditionalGenerator,
    }
    
    # Model registry for discriminators
    DISCRIMINATOR_REGISTRY = {
        "simple": SimpleDiscriminator,
        "complex": ComplexDiscriminator,
        "skip": SkipDiscriminator,
        "monochrome": MonochromeDiscriminator,
        "autoencoder": AutoencoderDiscriminator,
        "spectral_norm": SpectralNormDiscriminator,
        "progressive": ProgressiveDiscriminator,
        "multiscale": MultiScaleDiscriminator,
        "conditional": ConditionalDiscriminator,
    }
    
    @staticmethod
    def create_generator(model_type: str = "skip", void_dim: int = 64, noise_dim: int = 100, 
                        color_mode: int = 1, sparse: bool = False, device: str = "cuda", 
                        **kwargs) -> nn.Module:
        """
        Create a generator model based on the specified type.
        
        Args:
            model_type: Type of model ('simple', 'complex', 'skip', 'monochrome', 'autoencoder', 'progressive', 'conditional')
            void_dim: Dimension of the void/volume space
            noise_dim: Dimension of the noise input vector
            color_mode: 0 for monochrome, 1 for color
            sparse: Whether to use sparse convolutions
            device: Device to place the model on
            **kwargs: Additional arguments for specific model types
            
        Returns:
            Generator model
        """
        if model_type not in PyTorchModelFactory.GENERATOR_REGISTRY:
            print(f"Unknown generator type: {model_type}, defaulting to skip")
            model_type = "skip"
        
        generator_class = PyTorchModelFactory.GENERATOR_REGISTRY[model_type]
        
        # Handle special cases for different model types
        if model_type == "progressive":
            model = generator_class(
                void_dim=void_dim, 
                noise_dim=noise_dim, 
                color_mode=color_mode,
                max_resolution=kwargs.get("max_resolution", 128),
                sparse=sparse
            )
        elif model_type == "conditional":
            model = generator_class(
                void_dim=void_dim,
                noise_dim=noise_dim,
                condition_dim=kwargs.get("condition_dim", 10),
                color_mode=color_mode,
                sparse=sparse
            )
        else:
            model = generator_class(
                void_dim=void_dim,
                noise_dim=noise_dim,
                color_mode=color_mode,
                sparse=sparse
            )
        
        return model.to(device)
    
    @staticmethod
    def create_discriminator(model_type: str = "skip", void_dim: int = 64, noise_dim: int = 100,
                           color_mode: int = 1, sparse: bool = False, device: str = "cuda",
                           **kwargs) -> nn.Module:
        """
        Create a discriminator model based on the specified type.
        
        Args:
            model_type: Type of model ('simple', 'complex', 'skip', 'monochrome', 'autoencoder', 'spectral_norm', 'progressive', 'multiscale', 'conditional')
            void_dim: Dimension of the void/volume space
            noise_dim: Dimension of the noise input vector
            color_mode: 0 for monochrome, 1 for color
            sparse: Whether to use sparse convolutions
            device: Device to place the model on
            **kwargs: Additional arguments for specific model types
            
        Returns:
            Discriminator model
        """
        if model_type not in PyTorchModelFactory.DISCRIMINATOR_REGISTRY:
            print(f"Unknown discriminator type: {model_type}, defaulting to skip")
            model_type = "skip"
        
        discriminator_class = PyTorchModelFactory.DISCRIMINATOR_REGISTRY[model_type]
        
        # Handle special cases for different model types
        if model_type == "progressive":
            model = discriminator_class(
                void_dim=void_dim,
                noise_dim=noise_dim,
                color_mode=color_mode,
                max_resolution=kwargs.get("max_resolution", 128),
                sparse=sparse
            )
        elif model_type == "multiscale":
            model = discriminator_class(
                void_dim=void_dim,
                noise_dim=noise_dim,
                color_mode=color_mode,
                num_scales=kwargs.get("num_scales", 3),
                sparse=sparse
            )
        elif model_type == "conditional":
            model = discriminator_class(
                void_dim=void_dim,
                noise_dim=noise_dim,
                condition_dim=kwargs.get("condition_dim", 10),
                color_mode=color_mode,
                sparse=sparse
            )
        else:
            model = discriminator_class(
                void_dim=void_dim,
                noise_dim=noise_dim,
                color_mode=color_mode,
                sparse=sparse
            )
        
        return model.to(device)
    
    @staticmethod
    def create_gan_pair(model_type: str = "skip", void_dim: int = 64, noise_dim: int = 100,
                       color_mode: int = 1, sparse: bool = False, device: str = "cuda",
                       **kwargs) -> Tuple[nn.Module, nn.Module]:
        """
        Create a matched generator-discriminator pair.
        
        Args:
            model_type: Type of model
            void_dim: Dimension of the void/volume space
            noise_dim: Dimension of the noise input vector
            color_mode: 0 for monochrome, 1 for color
            sparse: Whether to use sparse convolutions
            device: Device to place the models on
            **kwargs: Additional arguments for specific model types
            
        Returns:
            Tuple of (generator, discriminator)
        """
        generator = PyTorchModelFactory.create_generator(
            model_type, void_dim, noise_dim, color_mode, sparse, device, **kwargs
        )
        discriminator = PyTorchModelFactory.create_discriminator(
            model_type, void_dim, noise_dim, color_mode, sparse, device, **kwargs
        )
        
        return generator, discriminator
    
    @staticmethod
    def get_model_info(model_type: str) -> Dict[str, Any]:
        """
        Get information about a specific model type.
        
        Args:
            model_type: Type of model to get info for
            
        Returns:
            Dictionary with model information
        """
        info = {
            "simple": {
                "description": "Basic 3D CNN architecture with transposed convolutions",
                "complexity": "low",
                "memory_usage": "low",
                "training_stability": "medium"
            },
            "complex": {
                "description": "Enhanced architecture with skip connections and attention",
                "complexity": "high",
                "memory_usage": "high",
                "training_stability": "high"
            },
            "skip": {
                "description": "U-Net style generator with skip connections",
                "complexity": "medium",
                "memory_usage": "medium",
                "training_stability": "high"
            },
            "monochrome": {
                "description": "Specialized for single-channel 3D data",
                "complexity": "low",
                "memory_usage": "low",
                "training_stability": "medium"
            },
            "autoencoder": {
                "description": "Autoencoder-based architecture",
                "complexity": "medium",
                "memory_usage": "medium",
                "training_stability": "medium"
            },
            "progressive": {
                "description": "Progressive growing for high-resolution 3D data",
                "complexity": "very_high",
                "memory_usage": "variable",
                "training_stability": "high"
            },
            "conditional": {
                "description": "Conditional generation with controllable outputs",
                "complexity": "high",
                "memory_usage": "high",
                "training_stability": "medium"
            }
        }
        
        return info.get(model_type, {"description": "Unknown model type"})
    
    @staticmethod
    def recommend_model(data_characteristics: Dict[str, Any]) -> str:
        """
        Recommend a model type based on data characteristics.
        
        Args:
            data_characteristics: Dictionary with data info (resolution, complexity, memory_constraints, etc.)
            
        Returns:
            Recommended model type
        """
        resolution = data_characteristics.get("resolution", 64)
        complexity = data_characteristics.get("complexity", "medium")
        memory_constraints = data_characteristics.get("memory_constraints", False)
        requires_conditioning = data_characteristics.get("requires_conditioning", False)
        
        # Conditional models for controlled generation
        if requires_conditioning:
            return "conditional"
        
        # High resolution data
        if resolution > 128:
            return "progressive"
        
        # Memory constrained environments
        if memory_constraints:
            return "simple" if complexity == "low" else "monochrome"
        
        # Default recommendations based on complexity
        if complexity == "low":
            return "simple"
        elif complexity == "high":
            return "complex"
        else:
            return "skip"  # Good balance for most cases
    
    @staticmethod
    def list_available_models() -> Dict[str, list]:
        """
        List all available model types.
        
        Returns:
            Dictionary with generator and discriminator model types
        """
        return {
            "generators": list(PyTorchModelFactory.GENERATOR_REGISTRY.keys()),
            "discriminators": list(PyTorchModelFactory.DISCRIMINATOR_REGISTRY.keys())
        }
    
    @staticmethod
    def validate_model_config(config: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate model configuration parameters.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        required_params = ["model_type", "void_dim", "noise_dim", "color_mode"]
        
        # Check required parameters
        for param in required_params:
            if param not in config:
                return False, f"Missing required parameter: {param}"
        
        # Validate parameter values
        if config["void_dim"] not in [32, 64, 128, 256]:
            return False, "void_dim must be one of [32, 64, 128, 256]"
        
        if config["noise_dim"] < 1:
            return False, "noise_dim must be positive"
        
        if config["color_mode"] not in [0, 1]:
            return False, "color_mode must be 0 (monochrome) or 1 (color)"
        
        if config["model_type"] not in PyTorchModelFactory.GENERATOR_REGISTRY:
            return False, f"Unknown model_type: {config['model_type']}"
        
        return True, "Configuration is valid"


class ModelArchitectureSearch:
    """Automated model architecture search and optimization."""
    
    def __init__(self, search_space: Dict[str, Any], evaluation_metric: str = "fid"):
        self.search_space = search_space
        self.evaluation_metric = evaluation_metric
        self.search_history = []
    
    def search(self, num_trials: int = 10, dataset=None) -> Dict[str, Any]:
        """
        Perform architecture search.
        
        Args:
            num_trials: Number of architectures to try
            dataset: Dataset for evaluation
            
        Returns:
            Best architecture configuration
        """
        best_config = None
        best_score = float('inf') if self.evaluation_metric in ['fid', 'loss'] else float('-inf')
        
        for trial in range(num_trials):
            # Sample configuration from search space
            config = self._sample_config()
            
            # Evaluate architecture
            score = self._evaluate_architecture(config, dataset)
            
            # Update best configuration
            is_better = (score < best_score if self.evaluation_metric in ['fid', 'loss'] 
                        else score > best_score)
            
            if is_better:
                best_score = score
                best_config = config
            
            self.search_history.append({
                "trial": trial,
                "config": config,
                "score": score
            })
        
        return best_config
    
    def _sample_config(self) -> Dict[str, Any]:
        """Sample a configuration from the search space."""
        import random
        
        config = {}
        for param, values in self.search_space.items():
            if isinstance(values, list):
                config[param] = random.choice(values)
            elif isinstance(values, dict) and "min" in values and "max" in values:
                config[param] = random.randint(values["min"], values["max"])
            else:
                config[param] = values
        
        return config
    
    def _evaluate_architecture(self, config: Dict[str, Any], dataset) -> float:
        """
        Evaluate an architecture configuration.
        
        Args:
            config: Architecture configuration
            dataset: Dataset for evaluation
            
        Returns:
            Evaluation score
        """
        # This is a simplified evaluation - in practice, you'd train the model
        # and evaluate on validation data
        
        # For now, return a mock score based on model complexity
        complexity_score = self._calculate_complexity(config)
        
        # Add some randomness to simulate actual evaluation
        import random
        noise = random.uniform(-0.1, 0.1)
        
        return complexity_score + noise
    
    def _calculate_complexity(self, config: Dict[str, Any]) -> float:
        """Calculate model complexity score."""
        complexity_weights = {
            "simple": 1.0,
            "complex": 3.0,
            "skip": 2.0,
            "monochrome": 1.0,
            "autoencoder": 2.0,
            "progressive": 4.0,
            "conditional": 3.5
        }
        
        base_complexity = complexity_weights.get(config.get("model_type", "skip"), 2.0)
        
        # Adjust for other parameters
        void_dim_factor = config.get("void_dim", 64) / 64.0
        noise_dim_factor = config.get("noise_dim", 100) / 100.0
        
        return base_complexity * void_dim_factor * noise_dim_factor


class ModelPlugin:
    """Base class for custom model architecture plugins."""
    
    def __init__(self, name: str):
        self.name = name
    
    def create_generator(self, **kwargs) -> nn.Module:
        """Create custom generator."""
        raise NotImplementedError("Subclasses must implement create_generator")
    
    def create_discriminator(self, **kwargs) -> nn.Module:
        """Create custom discriminator."""
        raise NotImplementedError("Subclasses must implement create_discriminator")
    
    def get_info(self) -> Dict[str, Any]:
        """Get plugin information."""
        return {
            "name": self.name,
            "description": "Custom model plugin",
            "version": "1.0.0"
        }


class PluginManager:
    """Manager for custom model architecture plugins."""
    
    def __init__(self):
        self.plugins = {}
    
    def register_plugin(self, plugin: ModelPlugin):
        """Register a custom model plugin."""
        self.plugins[plugin.name] = plugin
        
        # Add to factory registries
        PyTorchModelFactory.GENERATOR_REGISTRY[plugin.name] = plugin.create_generator
        PyTorchModelFactory.DISCRIMINATOR_REGISTRY[plugin.name] = plugin.create_discriminator
    
    def unregister_plugin(self, name: str):
        """Unregister a plugin."""
        if name in self.plugins:
            del self.plugins[name]
            
            # Remove from factory registries
            if name in PyTorchModelFactory.GENERATOR_REGISTRY:
                del PyTorchModelFactory.GENERATOR_REGISTRY[name]
            if name in PyTorchModelFactory.DISCRIMINATOR_REGISTRY:
                del PyTorchModelFactory.DISCRIMINATOR_REGISTRY[name]
    
    def list_plugins(self) -> Dict[str, Dict[str, Any]]:
        """List all registered plugins."""
        return {name: plugin.get_info() for name, plugin in self.plugins.items()}


# Backward compatibility functions
def create_generator(model_type: str = "skip", void_dim: int = 64, noise_dim: int = 100, 
                    color_mode: int = 1, sparse: bool = False, device: str = "cuda") -> nn.Module:
    """Backward compatible generator creation function."""
    return PyTorchModelFactory.create_generator(model_type, void_dim, noise_dim, color_mode, sparse, device)


def create_discriminator(model_type: str = "skip", void_dim: int = 64, noise_dim: int = 100,
                        color_mode: int = 1, sparse: bool = False, device: str = "cuda") -> nn.Module:
    """Backward compatible discriminator creation function."""
    return PyTorchModelFactory.create_discriminator(model_type, void_dim, noise_dim, color_mode, sparse, device)


# Global plugin manager instance
plugin_manager = PluginManager()


# Model utilities
class ModelUtils:
    """Utility functions for model operations."""
    
    @staticmethod
    def count_parameters(model: nn.Module) -> int:
        """Count the number of trainable parameters in a model."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    @staticmethod
    def get_model_size(model: nn.Module) -> float:
        """Get model size in MB."""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    
    @staticmethod
    def estimate_memory_usage(model: nn.Module, input_shape: Tuple[int, ...], 
                            batch_size: int = 1, device: str = "cuda") -> Dict[str, float]:
        """
        Estimate memory usage for model inference.
        
        Args:
            model: PyTorch model
            input_shape: Input tensor shape (without batch dimension)
            batch_size: Batch size
            device: Device type
            
        Returns:
            Dictionary with memory usage estimates in MB
        """
        # Create dummy input
        dummy_input = torch.randn(batch_size, *input_shape).to(device)
        
        # Measure memory before
        if device == "cuda":
            torch.cuda.empty_cache()
            mem_before = torch.cuda.memory_allocated() / 1024 / 1024
        else:
            mem_before = 0
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            _ = model(dummy_input)
        
        # Measure memory after
        if device == "cuda":
            mem_after = torch.cuda.memory_allocated() / 1024 / 1024
            torch.cuda.empty_cache()
        else:
            mem_after = 0
        
        model_size = ModelUtils.get_model_size(model)
        activation_memory = mem_after - mem_before - model_size
        
        return {
            "model_size_mb": model_size,
            "activation_memory_mb": max(0, activation_memory),
            "total_memory_mb": model_size + max(0, activation_memory)
        }
    
    @staticmethod
    def compare_models(model1: nn.Module, model2: nn.Module) -> Dict[str, Any]:
        """
        Compare two models in terms of parameters and size.
        
        Args:
            model1: First model
            model2: Second model
            
        Returns:
            Comparison results
        """
        params1 = ModelUtils.count_parameters(model1)
        params2 = ModelUtils.count_parameters(model2)
        
        size1 = ModelUtils.get_model_size(model1)
        size2 = ModelUtils.get_model_size(model2)
        
        return {
            "model1_params": params1,
            "model2_params": params2,
            "param_ratio": params2 / params1 if params1 > 0 else float('inf'),
            "model1_size_mb": size1,
            "model2_size_mb": size2,
            "size_ratio": size2 / size1 if size1 > 0 else float('inf')
        }

# ============================================================================
# ADDITIONAL SPARSE-AWARE LAYERS
# ============================================================================

class SparseActivation(nn.Module):
    """
    Sparse-aware activation function that preserves sparsity patterns.
    """
    
    def __init__(self, activation: str = "relu", sparse_threshold: float = 0.1,
                 preserve_sparsity: bool = True):
        super().__init__()
        self.activation_name = activation
        self.sparse_threshold = sparse_threshold
        self.preserve_sparsity = preserve_sparsity
        
        # Initialize activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.2)
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_sparse and self.preserve_sparsity:
            # Apply activation only to non-zero values
            indices = x._indices()
            values = x._values()
            shape = x.shape
            
            # Apply activation to values
            activated_values = self.activation(values)
            
            # Create new sparse tensor
            return torch.sparse.FloatTensor(indices, activated_values, shape)
        else:
            # Standard activation
            out = self.activation(x)
            
            # Convert to sparse if beneficial
            if not x.is_sparse and self.preserve_sparsity:
                sparsity = (out == 0).float().mean().item()
                if sparsity > self.sparse_threshold:
                    return out.to_sparse()
            
            return out


class SparseDropout3d(nn.Module):
    """
    3D dropout layer optimized for sparse tensors.
    """
    
    def __init__(self, p: float = 0.5, inplace: bool = False, sparse_aware: bool = True):
        super().__init__()
        self.p = p
        self.inplace = inplace
        self.sparse_aware = sparse_aware
        self.dropout = nn.Dropout3d(p=p, inplace=inplace)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_sparse and self.sparse_aware:
            # Apply dropout only to non-zero values for efficiency
            if not self.training:
                return x
            
            indices = x._indices()
            values = x._values()
            shape = x.shape
            
            # Apply dropout to values
            if self.p > 0:
                # Create dropout mask for values only
                keep_prob = 1 - self.p
                mask = torch.bernoulli(torch.full_like(values, keep_prob))
                dropped_values = values * mask / keep_prob
            else:
                dropped_values = values
            
            return torch.sparse.FloatTensor(indices, dropped_values, shape)
        else:
            return self.dropout(x)


class SparsePooling3d(nn.Module):
    """
    3D pooling layer that handles sparse tensors efficiently.
    """
    
    def __init__(self, kernel_size: int = 2, stride: int = 2, padding: int = 0,
                 pool_type: str = "max", sparse_threshold: float = 0.1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.pool_type = pool_type
        self.sparse_threshold = sparse_threshold
        
        if pool_type == "max":
            self.pool = nn.MaxPool3d(kernel_size, stride, padding)
        elif pool_type == "avg":
            self.pool = nn.AvgPool3d(kernel_size, stride, padding)
        elif pool_type == "adaptive_max":
            self.pool = nn.AdaptiveMaxPool3d(kernel_size)
        elif pool_type == "adaptive_avg":
            self.pool = nn.AdaptiveAvgPool3d(kernel_size)
        else:
            raise ValueError(f"Unsupported pool type: {pool_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_sparse:
            # Convert to dense for pooling, then back to sparse if beneficial
            x_dense = x.to_dense()
            out = self.pool(x_dense)
            
            # Check if output should remain sparse
            sparsity = (out == 0).float().mean().item()
            if sparsity > self.sparse_threshold:
                return out.to_sparse()
            return out
        else:
            return self.pool(x)


class SparseUpsampling3d(nn.Module):
    """
    3D upsampling layer optimized for sparse tensors.
    """
    
    def __init__(self, scale_factor: int = 2, mode: str = "nearest",
                 sparse_threshold: float = 0.1, preserve_sparsity: bool = True):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.sparse_threshold = sparse_threshold
        self.preserve_sparsity = preserve_sparsity
        
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_sparse:
            # Convert to dense for upsampling
            x_dense = x.to_dense()
            out = self.upsample(x_dense)
            
            # Maintain sparsity if beneficial
            if self.preserve_sparsity:
                sparsity = (out == 0).float().mean().item()
                if sparsity > self.sparse_threshold:
                    return out.to_sparse()
            return out
        else:
            out = self.upsample(x)
            
            # Convert to sparse if beneficial
            if self.preserve_sparsity:
                sparsity = (out == 0).float().mean().item()
                if sparsity > self.sparse_threshold:
                    return out.to_sparse()
            return out


class SparseLinear(nn.Module):
    """
    Linear layer optimized for sparse inputs.
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 sparse_threshold: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.sparse_threshold = sparse_threshold
        self.in_features = in_features
        self.out_features = out_features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_sparse:
            # For sparse inputs, convert to dense for linear operation
            x_dense = x.to_dense()
            out = self.linear(x_dense)
            
            # Check if output should be sparse
            sparsity = (out == 0).float().mean().item()
            if sparsity > self.sparse_threshold:
                return out.to_sparse()
            return out
        else:
            return self.linear(x)


class SparseLayerNorm3d(nn.Module):
    """
    3D Layer normalization for sparse tensors.
    """
    
    def __init__(self, normalized_shape: Union[int, List[int]], eps: float = 1e-5,
                 elementwise_affine: bool = True, sparse_mode: str = "auto"):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = [normalized_shape]
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.sparse_mode = sparse_mode
        
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_sparse:
            if self.sparse_mode == "preserve":
                return self._sparse_layer_norm(x)
            else:
                # Convert to dense for layer norm
                x_dense = x.to_dense()
                out = F.layer_norm(x_dense, self.normalized_shape, self.weight, self.bias, self.eps)
                
                if self.sparse_mode == "auto":
                    sparsity = (out == 0).float().mean().item()
                    if sparsity > 0.1:
                        return out.to_sparse()
                return out
        else:
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    
    def _sparse_layer_norm(self, x: torch.sparse.FloatTensor) -> torch.sparse.FloatTensor:
        """Apply layer normalization to sparse tensor values only."""
        indices = x._indices()
        values = x._values()
        shape = x.shape
        
        # Normalize values
        mean = values.mean()
        var = values.var(unbiased=False)
        normalized_values = (values - mean) / torch.sqrt(var + self.eps)
        
        # Apply affine transformation
        if self.elementwise_affine:
            # For simplicity, apply same weight/bias to all values
            # In practice, you might want channel-wise normalization
            normalized_values = normalized_values * self.weight.mean() + self.bias.mean()
        
        return torch.sparse.FloatTensor(indices, normalized_values, shape)


class SparseSequential(nn.Sequential):
    """
    Sequential container optimized for sparse tensor propagation.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sparsity_tracking = True
        self.layer_sparsities = []
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.sparsity_tracking:
            self.layer_sparsities = []
        
        for i, module in enumerate(self):
            x = module(x)
            
            if self.sparsity_tracking:
                if x.is_sparse:
                    sparsity = 1.0 - (x._nnz() / x.numel())
                else:
                    sparsity = (x == 0).float().mean().item()
                self.layer_sparsities.append(sparsity)
        
        return x
    
    def get_sparsity_profile(self) -> List[float]:
        """Get sparsity at each layer."""
        return self.layer_sparsities.copy()
    
    def optimize_sparsity_thresholds(self):
        """Optimize sparsity thresholds based on observed patterns."""
        for i, (module, sparsity) in enumerate(zip(self, self.layer_sparsities)):
            if hasattr(module, 'sparse_threshold'):
                # Adjust threshold based on observed sparsity
                if sparsity > 0.8:
                    module.sparse_threshold = max(0.05, module.sparse_threshold * 0.9)
                elif sparsity < 0.2:
                    module.sparse_threshold = min(0.5, module.sparse_threshold * 1.1)


# ============================================================================
# SPARSE TENSOR UTILITIES FOR MODELS
# ============================================================================

class SparseModelWrapper(nn.Module):
    """
    Wrapper that automatically converts a regular model to work with sparse tensors.
    """
    
    def __init__(self, model: nn.Module, auto_convert: bool = True, 
                 sparse_threshold: float = 0.1):
        super().__init__()
        self.model = model
        self.auto_convert = auto_convert
        self.sparse_threshold = sparse_threshold
        
        # Replace layers with sparse equivalents if beneficial
        if auto_convert:
            self._convert_to_sparse_layers()
    
    def _convert_to_sparse_layers(self):
        """Convert regular layers to sparse-aware versions."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv3d):
                # Replace with SparseConv3d
                sparse_conv = SparseConv3d(
                    module.in_channels, module.out_channels, module.kernel_size[0],
                    module.stride[0], module.padding[0], module.bias is not None,
                    self.sparse_threshold
                )
                # Copy weights
                sparse_conv.conv.weight.data = module.weight.data
                if module.bias is not None:
                    sparse_conv.conv.bias.data = module.bias.data
                
                # Replace in model
                self._replace_module(name, sparse_conv)
            
            elif isinstance(module, nn.ConvTranspose3d):
                # Replace with SparseConvTranspose3d
                sparse_conv_t = SparseConvTranspose3d(
                    module.in_channels, module.out_channels, module.kernel_size[0],
                    module.stride[0], module.padding[0], module.output_padding[0],
                    module.bias is not None, self.sparse_threshold
                )
                # Copy weights
                sparse_conv_t.conv_transpose.weight.data = module.weight.data
                if module.bias is not None:
                    sparse_conv_t.conv_transpose.bias.data = module.bias.data
                
                # Replace in model
                self._replace_module(name, sparse_conv_t)
            
            elif isinstance(module, nn.BatchNorm3d):
                # Replace with SparseBatchNorm3d
                sparse_bn = SparseBatchNorm3d(
                    module.num_features, module.eps, module.momentum,
                    module.affine, module.track_running_stats
                )
                # Copy parameters
                sparse_bn.bn.weight.data = module.weight.data
                sparse_bn.bn.bias.data = module.bias.data
                sparse_bn.bn.running_mean.data = module.running_mean.data
                sparse_bn.bn.running_var.data = module.running_var.data
                
                # Replace in model
                self._replace_module(name, sparse_bn)
    
    def _replace_module(self, name: str, new_module: nn.Module):
        """Replace a module in the model."""
        parts = name.split('.')
        parent = self.model
        
        for part in parts[:-1]:
            parent = getattr(parent, part)
        
        setattr(parent, parts[-1], new_module)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def get_sparsity_info(self) -> Dict[str, Any]:
        """Get information about sparsity in the model."""
        info = {
            "total_sparse_layers": 0,
            "layer_info": {}
        }
        
        for name, module in self.model.named_modules():
            if hasattr(module, 'get_sparsity_stats'):
                info["total_sparse_layers"] += 1
                info["layer_info"][name] = module.get_sparsity_stats()
        
        return info


def convert_model_to_sparse(model: nn.Module, sparse_threshold: float = 0.1) -> nn.Module:
    """
    Convert a regular model to use sparse-aware layers.
    
    Args:
        model: Original model
        sparse_threshold: Threshold for sparsity conversion
        
    Returns:
        Model with sparse-aware layers
    """
    return SparseModelWrapper(model, auto_convert=True, sparse_threshold=sparse_threshold)


def analyze_model_sparsity(model: nn.Module, sample_input: torch.Tensor) -> Dict[str, Any]:
    """
    Analyze sparsity patterns in a model.
    
    Args:
        model: Model to analyze
        sample_input: Sample input tensor
        
    Returns:
        Sparsity analysis results
    """
    model.eval()
    sparsity_info = {}
    
    # Hook to capture intermediate activations
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                if output.is_sparse:
                    sparsity = 1.0 - (output._nnz() / output.numel())
                else:
                    sparsity = (output == 0).float().mean().item()
                
                sparsity_info[name] = {
                    "sparsity": sparsity,
                    "is_sparse": output.is_sparse,
                    "shape": tuple(output.shape),
                    "memory_usage": output.element_size() * output.numel()
                }
        return hook
    
    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            hook = module.register_forward_hook(hook_fn(name))
            hooks.append(hook)
    
    # Forward pass
    with torch.no_grad():
        _ = model(sample_input)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Calculate summary statistics
    sparsities = [info["sparsity"] for info in sparsity_info.values()]
    total_memory = sum(info["memory_usage"] for info in sparsity_info.values())
    
    summary = {
        "layer_sparsities": sparsity_info,
        "average_sparsity": sum(sparsities) / len(sparsities) if sparsities else 0,
        "max_sparsity": max(sparsities) if sparsities else 0,
        "min_sparsity": min(sparsities) if sparsities else 0,
        "total_memory_usage": total_memory,
        "num_layers_analyzed": len(sparsity_info)
    }
    
    return summary
# ============================================================================
# ADVANCED SPARSE 3D CONVOLUTION LAYERS
# ============================================================================

class AdvancedSparseConv3d(nn.Module):
    """
    Advanced 3D convolution layer with optimized sparse tensor operations.
    
    This implementation provides more efficient sparse convolution by:
    1. Using sparse tensor operations where possible
    2. Implementing custom sparse convolution kernels
    3. Optimizing memory usage for sparse inputs
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, dilation: int = 1, groups: int = 1,
                 bias: bool = True, sparse_threshold: float = 0.1, 
                 sparse_algorithm: str = "auto"):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.sparse_threshold = sparse_threshold
        self.sparse_algorithm = sparse_algorithm  # "auto", "dense", "sparse", "hybrid"
        
        # Standard convolution for fallback
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, 
                             padding, dilation, groups, bias)
        
        # Performance tracking
        self.sparse_ops_count = 0
        self.dense_ops_count = 0
        self.hybrid_ops_count = 0
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_sparse and self.sparse_algorithm in ["auto", "sparse", "hybrid"]:
            return self._sparse_forward(x)
        else:
            self.dense_ops_count += 1
            out = self.conv(x)
            
            # Convert to sparse if output is sparse enough
            if self.sparse_algorithm == "auto":
                sparsity = (out == 0).float().mean().item()
                if sparsity > self.sparse_threshold:
                    return out.to_sparse()
            
            return out
    
    def _sparse_forward(self, x: torch.sparse.FloatTensor) -> torch.Tensor:
        """Optimized forward pass for sparse tensors."""
        if self.sparse_algorithm == "sparse":
            return self._pure_sparse_conv(x)
        elif self.sparse_algorithm == "hybrid":
            return self._hybrid_sparse_conv(x)
        else:  # auto
            # Choose best algorithm based on sparsity
            sparsity = 1.0 - (x._nnz() / x.numel())
            if sparsity > 0.8:
                return self._pure_sparse_conv(x)
            else:
                return self._hybrid_sparse_conv(x)
    
    def _pure_sparse_conv(self, x: torch.sparse.FloatTensor) -> torch.Tensor:
        """
        Pure sparse convolution implementation.
        
        This method performs convolution only on non-zero elements,
        which is more efficient for very sparse tensors.
        """
        self.sparse_ops_count += 1
        
        # Get sparse tensor components
        indices = x._indices()  # Shape: [ndim, nnz]
        values = x._values()    # Shape: [nnz]
        shape = x.shape
        
        # For now, fall back to dense convolution
        # In a full implementation, you would implement sparse convolution kernels
        x_dense = x.to_dense()
        out = self.conv(x_dense)
        
        # Convert back to sparse if beneficial
        sparsity = (out == 0).float().mean().item()
        if sparsity > self.sparse_threshold:
            return out.to_sparse()
        
        return out
    
    def _hybrid_sparse_conv(self, x: torch.sparse.FloatTensor) -> torch.Tensor:
        """
        Hybrid sparse convolution that processes sparse regions efficiently.
        
        This method identifies dense regions and processes them separately
        from sparse regions for optimal performance.
        """
        self.hybrid_ops_count += 1
        
        # Convert to dense for now - in practice, you'd implement
        # region-based processing
        x_dense = x.to_dense()
        out = self.conv(x_dense)
        
        # Maintain sparsity structure
        sparsity = (out == 0).float().mean().item()
        if sparsity > self.sparse_threshold:
            return out.to_sparse()
        
        return out
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for this layer."""
        total_ops = self.sparse_ops_count + self.dense_ops_count + self.hybrid_ops_count
        
        return {
            "sparse_ops_ratio": self.sparse_ops_count / max(total_ops, 1),
            "dense_ops_ratio": self.dense_ops_count / max(total_ops, 1),
            "hybrid_ops_ratio": self.hybrid_ops_count / max(total_ops, 1),
            "total_operations": total_ops,
            "sparse_algorithm": self.sparse_algorithm,
            "sparse_threshold": self.sparse_threshold,
        }
    
    def optimize_algorithm(self):
        """Optimize the sparse algorithm based on observed patterns."""
        stats = self.get_performance_stats()
        
        if stats["sparse_ops_ratio"] > 0.8:
            # Mostly sparse operations - use pure sparse
            self.sparse_algorithm = "sparse"
        elif stats["dense_ops_ratio"] > 0.8:
            # Mostly dense operations - use dense
            self.sparse_algorithm = "dense"
        else:
            # Mixed - use hybrid
            self.sparse_algorithm = "hybrid"


class AdvancedSparseConvTranspose3d(nn.Module):
    """
    Advanced 3D transposed convolution with optimized sparse operations.
    
    This layer is optimized for generator networks where sparse representations
    can significantly reduce memory usage during upsampling.
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, output_padding: int = 0,
                 dilation: int = 1, groups: int = 1, bias: bool = True,
                 sparse_threshold: float = 0.1, sparse_algorithm: str = "auto",
                 memory_efficient: bool = True):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.sparse_threshold = sparse_threshold
        self.sparse_algorithm = sparse_algorithm
        self.memory_efficient = memory_efficient
        
        # Standard transposed convolution
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride, padding,
            output_padding, groups, bias, dilation
        )
        
        # Performance tracking
        self.upsampling_stats = {
            "sparse_inputs": 0,
            "dense_inputs": 0,
            "memory_savings": [],
            "output_sparsities": []
        }
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_sparse:
            self.upsampling_stats["sparse_inputs"] += 1
            return self._sparse_transpose_forward(x)
        else:
            self.upsampling_stats["dense_inputs"] += 1
            out = self.conv_transpose(x)
            
            # Convert to sparse if beneficial
            if self.sparse_algorithm == "auto":
                sparsity = (out == 0).float().mean().item()
                self.upsampling_stats["output_sparsities"].append(sparsity)
                
                if sparsity > self.sparse_threshold:
                    sparse_out = out.to_sparse()
                    
                    if self.memory_efficient:
                        # Calculate memory savings
                        dense_memory = self._estimate_memory(out)
                        sparse_memory = self._estimate_memory(sparse_out)
                        savings = (dense_memory - sparse_memory) / dense_memory
                        self.upsampling_stats["memory_savings"].append(savings)
                    
                    return sparse_out
            
            return out
    
    def _sparse_transpose_forward(self, x: torch.sparse.FloatTensor) -> torch.Tensor:
        """Optimized transposed convolution for sparse inputs."""
        if self.sparse_algorithm == "sparse":
            return self._sparse_transpose_conv(x)
        else:
            # Convert to dense, process, then decide on output format
            x_dense = x.to_dense()
            out = self.conv_transpose(x_dense)
            
            sparsity = (out == 0).float().mean().item()
            self.upsampling_stats["output_sparsities"].append(sparsity)
            
            if sparsity > self.sparse_threshold:
                return out.to_sparse()
            
            return out
    
    def _sparse_transpose_conv(self, x: torch.sparse.FloatTensor) -> torch.Tensor:
        """
        Sparse transposed convolution implementation.
        
        This method optimizes transposed convolution for sparse inputs
        by processing only non-zero regions.
        """
        # For now, fall back to dense implementation
        # In practice, you would implement optimized sparse transpose convolution
        x_dense = x.to_dense()
        out = self.conv_transpose(x_dense)
        
        # Maintain sparsity if beneficial
        sparsity = (out == 0).float().mean().item()
        if sparsity > self.sparse_threshold:
            return out.to_sparse()
        
        return out
    
    def _estimate_memory(self, tensor: torch.Tensor) -> int:
        """Estimate memory usage of a tensor."""
        if tensor.is_sparse:
            indices_memory = tensor._indices().element_size() * tensor._indices().numel()
            values_memory = tensor._values().element_size() * tensor._values().numel()
            return indices_memory + values_memory
        else:
            return tensor.element_size() * tensor.numel()
    
    def get_upsampling_stats(self) -> Dict[str, Any]:
        """Get upsampling performance statistics."""
        total_inputs = (self.upsampling_stats["sparse_inputs"] + 
                       self.upsampling_stats["dense_inputs"])
        
        avg_sparsity = (sum(self.upsampling_stats["output_sparsities"]) / 
                       len(self.upsampling_stats["output_sparsities"]) 
                       if self.upsampling_stats["output_sparsities"] else 0)
        
        avg_memory_savings = (sum(self.upsampling_stats["memory_savings"]) / 
                             len(self.upsampling_stats["memory_savings"]) 
                             if self.upsampling_stats["memory_savings"] else 0)
        
        return {
            "sparse_input_ratio": self.upsampling_stats["sparse_inputs"] / max(total_inputs, 1),
            "average_output_sparsity": avg_sparsity,
            "average_memory_savings": avg_memory_savings,
            "total_forward_passes": total_inputs,
            "sparse_algorithm": self.sparse_algorithm,
        }


class SparseDepthwiseConv3d(nn.Module):
    """
    Sparse depthwise 3D convolution for efficient processing.
    
    Depthwise convolution processes each channel separately,
    which can be more efficient for sparse tensors.
    """
    
    def __init__(self, channels: int, kernel_size: int = 3, stride: int = 1,
                 padding: int = 1, bias: bool = True, sparse_threshold: float = 0.1):
        super().__init__()
        
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.sparse_threshold = sparse_threshold
        
        # Depthwise convolution (groups = channels)
        self.depthwise = nn.Conv3d(channels, channels, kernel_size, stride, 
                                  padding, groups=channels, bias=bias)
        
        # Channel-wise sparsity tracking
        self.channel_sparsities = torch.zeros(channels)
        self.forward_count = 0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.forward_count += 1
        
        if x.is_sparse:
            # Process sparse tensor
            x_dense = x.to_dense()
            out = self.depthwise(x_dense)
            
            # Track channel-wise sparsity
            self._update_channel_sparsities(out)
            
            # Convert back to sparse if beneficial
            sparsity = (out == 0).float().mean().item()
            if sparsity > self.sparse_threshold:
                return out.to_sparse()
            
            return out
        else:
            out = self.depthwise(x)
            
            # Track channel-wise sparsity
            self._update_channel_sparsities(out)
            
            # Convert to sparse if beneficial
            sparsity = (out == 0).float().mean().item()
            if sparsity > self.sparse_threshold:
                return out.to_sparse()
            
            return out
    
    def _update_channel_sparsities(self, x: torch.Tensor):
        """Update channel-wise sparsity statistics."""
        if len(x.shape) == 5:  # [batch, channels, depth, height, width]
            for c in range(self.channels):
                channel_data = x[:, c, :, :, :]
                channel_sparsity = (channel_data == 0).float().mean().item()
                
                # Exponential moving average
                alpha = 0.1
                self.channel_sparsities[c] = (
                    alpha * channel_sparsity + 
                    (1 - alpha) * self.channel_sparsities[c]
                )
    
    def get_channel_sparsity_info(self) -> Dict[str, Any]:
        """Get channel-wise sparsity information."""
        return {
            "channel_sparsities": self.channel_sparsities.tolist(),
            "average_sparsity": self.channel_sparsities.mean().item(),
            "max_sparsity": self.channel_sparsities.max().item(),
            "min_sparsity": self.channel_sparsities.min().item(),
            "forward_passes": self.forward_count,
        }


class SparseSeparableConv3d(nn.Module):
    """
    Sparse separable 3D convolution combining depthwise and pointwise convolutions.
    
    This can be more efficient than standard convolution for sparse tensors
    as it reduces the number of parameters and computations.
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, bias: bool = True,
                 sparse_threshold: float = 0.1):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sparse_threshold = sparse_threshold
        
        # Depthwise convolution
        self.depthwise = SparseDepthwiseConv3d(
            in_channels, kernel_size, stride, padding, bias, sparse_threshold
        )
        
        # Pointwise convolution (1x1x1)
        self.pointwise = AdvancedSparseConv3d(
            in_channels, out_channels, 1, 1, 0, bias=bias,
            sparse_threshold=sparse_threshold
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Depthwise convolution
        x = self.depthwise(x)
        
        # Pointwise convolution
        x = self.pointwise(x)
        
        return x
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for both components."""
        return {
            "depthwise_stats": self.depthwise.get_channel_sparsity_info(),
            "pointwise_stats": self.pointwise.get_performance_stats(),
        }


class SparseConvBlock3d(nn.Module):
    """
    Complete sparse convolution block with normalization and activation.
    
    This block combines convolution, normalization, and activation
    while maintaining sparsity throughout the operations.
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, bias: bool = False,
                 norm_type: str = "batch", activation: str = "relu",
                 sparse_threshold: float = 0.1, dropout_p: float = 0.0):
        super().__init__()
        
        self.sparse_threshold = sparse_threshold
        
        # Convolution
        self.conv = AdvancedSparseConv3d(
            in_channels, out_channels, kernel_size, stride, padding,
            bias=bias, sparse_threshold=sparse_threshold
        )
        
        # Normalization
        if norm_type == "batch":
            self.norm = SparseBatchNorm3d(out_channels)
        elif norm_type == "layer":
            self.norm = SparseLayerNorm3d([out_channels])
        else:
            self.norm = nn.Identity()
        
        # Activation
        self.activation = SparseActivation(activation, sparse_threshold)
        
        # Dropout
        if dropout_p > 0:
            self.dropout = SparseDropout3d(dropout_p)
        else:
            self.dropout = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x
    
    def get_block_stats(self) -> Dict[str, Any]:
        """Get statistics for the entire block."""
        stats = {
            "conv_stats": self.conv.get_performance_stats(),
        }
        
        if hasattr(self.norm, 'get_processing_stats'):
            stats["norm_stats"] = self.norm.get_processing_stats()
        
        return stats


# ============================================================================
# SPARSE CONVOLUTION UTILITIES
# ============================================================================

def create_sparse_conv_layer(layer_type: str, **kwargs) -> nn.Module:
    """
    Factory function to create sparse convolution layers.
    
    Args:
        layer_type: Type of layer ("conv", "conv_transpose", "depthwise", "separable", "block")
        **kwargs: Layer-specific arguments
        
    Returns:
        Sparse convolution layer
    """
    if layer_type == "conv":
        return AdvancedSparseConv3d(**kwargs)
    elif layer_type == "conv_transpose":
        return AdvancedSparseConvTranspose3d(**kwargs)
    elif layer_type == "depthwise":
        return SparseDepthwiseConv3d(**kwargs)
    elif layer_type == "separable":
        return SparseSeparableConv3d(**kwargs)
    elif layer_type == "block":
        return SparseConvBlock3d(**kwargs)
    else:
        raise ValueError(f"Unknown sparse layer type: {layer_type}")


def optimize_conv_layers_for_sparsity(model: nn.Module, 
                                     sparse_threshold: float = 0.1) -> nn.Module:
    """
    Replace standard convolution layers with sparse-aware versions.
    
    Args:
        model: Model to optimize
        sparse_threshold: Sparsity threshold for conversions
        
    Returns:
        Optimized model
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv3d) and not isinstance(module, (SparseConv3d, AdvancedSparseConv3d)):
            # Replace with advanced sparse conv
            sparse_conv = AdvancedSparseConv3d(
                module.in_channels, module.out_channels, module.kernel_size[0],
                module.stride[0], module.padding[0], module.dilation[0],
                module.groups, module.bias is not None, sparse_threshold
            )
            
            # Copy weights
            sparse_conv.conv.weight.data = module.weight.data
            if module.bias is not None:
                sparse_conv.conv.bias.data = module.bias.data
            
            # Replace in model
            _replace_module_in_model(model, name, sparse_conv)
        
        elif isinstance(module, nn.ConvTranspose3d) and not isinstance(module, (SparseConvTranspose3d, AdvancedSparseConvTranspose3d)):
            # Replace with advanced sparse conv transpose
            sparse_conv_t = AdvancedSparseConvTranspose3d(
                module.in_channels, module.out_channels, module.kernel_size[0],
                module.stride[0], module.padding[0], module.output_padding[0],
                module.dilation[0], module.groups, module.bias is not None,
                sparse_threshold
            )
            
            # Copy weights
            sparse_conv_t.conv_transpose.weight.data = module.weight.data
            if module.bias is not None:
                sparse_conv_t.conv_transpose.bias.data = module.bias.data
            
            # Replace in model
            _replace_module_in_model(model, name, sparse_conv_t)
    
    return model


def _replace_module_in_model(model: nn.Module, module_name: str, new_module: nn.Module):
    """Helper function to replace a module in a model."""
    parts = module_name.split('.')
    parent = model
    
    for part in parts[:-1]:
        parent = getattr(parent, part)
    
    setattr(parent, parts[-1], new_module)


def benchmark_sparse_convolutions(input_tensor: torch.Tensor, 
                                 layer_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Benchmark different sparse convolution implementations.
    
    Args:
        input_tensor: Input tensor for benchmarking
        layer_configs: List of layer configurations to benchmark
        
    Returns:
        Benchmark results
    """
    import time
    
    results = {}
    
    for i, config in enumerate(layer_configs):
        layer_name = config.pop("name", f"layer_{i}")
        layer_type = config.pop("type", "conv")
        
        # Create layer
        layer = create_sparse_conv_layer(layer_type, **config)
        layer.eval()
        
        # Warm up
        with torch.no_grad():
            for _ in range(5):
                _ = layer(input_tensor)
        
        # Benchmark
        torch.cuda.synchronize() if input_tensor.is_cuda else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(100):
                output = layer(input_tensor)
        
        torch.cuda.synchronize() if input_tensor.is_cuda else None
        end_time = time.time()
        
        # Calculate metrics
        avg_time = (end_time - start_time) / 100
        
        if hasattr(layer, 'get_performance_stats'):
            layer_stats = layer.get_performance_stats()
        else:
            layer_stats = {}
        
        results[layer_name] = {
            "average_time_ms": avg_time * 1000,
            "output_shape": tuple(output.shape),
            "output_is_sparse": output.is_sparse,
            "layer_stats": layer_stats,
        }
        
        if output.is_sparse:
            results[layer_name]["output_sparsity"] = 1.0 - (output._nnz() / output.numel())
        else:
            results[layer_name]["output_sparsity"] = (output == 0).float().mean().item()
    
    return results

# ============================================================================
# ADVANCED SPARSE NORMALIZATION AND ACTIVATION LAYERS
# ============================================================================

class AdvancedSparseBatchNorm3d(nn.Module):
    """
    Advanced sparse batch normalization with multiple processing modes.
    
    This implementation provides several strategies for normalizing sparse tensors:
    1. Dense conversion: Convert to dense, normalize, convert back
    2. Sparse-aware: Normalize only non-zero values
    3. Channel-wise sparse: Per-channel sparse normalization
    4. Adaptive: Choose strategy based on sparsity patterns
    """
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = True, track_running_stats: bool = True,
                 sparse_mode: str = "adaptive", sparsity_threshold: float = 0.5):
        super().__init__()
        
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.sparse_mode = sparse_mode
        self.sparsity_threshold = sparsity_threshold
        
        # Standard batch norm for fallback
        self.bn = nn.BatchNorm3d(num_features, eps, momentum, affine, track_running_stats)
        
        # Sparse-specific parameters
        if affine:
            self.sparse_weight = nn.Parameter(torch.ones(num_features))
            self.sparse_bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('sparse_weight', None)
            self.register_parameter('sparse_bias', None)
        
        # Running statistics for sparse normalization
        if track_running_stats:
            self.register_buffer('sparse_running_mean', torch.zeros(num_features))
            self.register_buffer('sparse_running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked_sparse', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('sparse_running_mean', None)
            self.register_parameter('sparse_running_var', None)
            self.register_parameter('num_batches_tracked_sparse', None)
        
        # Performance tracking
        self.mode_usage = {
            "dense": 0,
            "sparse_aware": 0,
            "channel_wise": 0,
            "adaptive": 0
        }
        
        # Channel-wise sparsity tracking
        self.channel_sparsities = torch.zeros(num_features)
        self.channel_update_count = 0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_sparse:
            self.mode_usage["dense"] += 1
            return self.bn(x)
        
        # Determine processing mode
        sparsity = 1.0 - (x._nnz() / x.numel())
        
        if self.sparse_mode == "dense":
            return self._dense_mode(x)
        elif self.sparse_mode == "sparse_aware":
            return self._sparse_aware_mode(x)
        elif self.sparse_mode == "channel_wise":
            return self._channel_wise_mode(x)
        elif self.sparse_mode == "adaptive":
            return self._adaptive_mode(x, sparsity)
        else:
            raise ValueError(f"Unknown sparse mode: {self.sparse_mode}")
    
    def _dense_mode(self, x: torch.sparse.FloatTensor) -> torch.Tensor:
        """Convert to dense, normalize, optionally convert back."""
        self.mode_usage["dense"] += 1
        
        x_dense = x.to_dense()
        out = self.bn(x_dense)
        
        # Convert back to sparse if beneficial
        sparsity = (out == 0).float().mean().item()
        if sparsity > self.sparsity_threshold:
            return out.to_sparse()
        
        return out
    
    def _sparse_aware_mode(self, x: torch.sparse.FloatTensor) -> torch.sparse.FloatTensor:
        """Normalize only non-zero values while preserving sparsity."""
        self.mode_usage["sparse_aware"] += 1
        
        indices = x._indices()
        values = x._values()
        shape = x.shape
        
        # Get channel indices for grouping
        if len(indices) >= 2:  # Assuming [batch, channel, ...]
            channel_indices = indices[1]
            
            normalized_values = torch.zeros_like(values)
            
            for c in range(self.num_features):
                channel_mask = channel_indices == c
                if not channel_mask.any():
                    continue
                
                channel_values = values[channel_mask]
                
                if self.training:
                    # Calculate statistics for non-zero values only
                    mean = channel_values.mean()
                    var = channel_values.var(unbiased=False)
                    
                    # Update running statistics
                    if self.track_running_stats:
                        with torch.no_grad():
                            self.sparse_running_mean[c] = (
                                (1 - self.momentum) * self.sparse_running_mean[c] + 
                                self.momentum * mean
                            )
                            self.sparse_running_var[c] = (
                                (1 - self.momentum) * self.sparse_running_var[c] + 
                                self.momentum * var
                            )
                else:
                    # Use running statistics
                    mean = self.sparse_running_mean[c]
                    var = self.sparse_running_var[c]
                
                # Normalize
                normalized_channel_values = (channel_values - mean) / torch.sqrt(var + self.eps)
                
                # Apply affine transformation
                if self.affine:
                    normalized_channel_values = (
                        normalized_channel_values * self.sparse_weight[c] + self.sparse_bias[c]
                    )
                
                normalized_values[channel_mask] = normalized_channel_values
        else:
            # Fallback for unexpected tensor structure
            mean = values.mean()
            var = values.var(unbiased=False)
            normalized_values = (values - mean) / torch.sqrt(var + self.eps)
        
        return torch.sparse.FloatTensor(indices, normalized_values, shape)
    
    def _channel_wise_mode(self, x: torch.sparse.FloatTensor) -> torch.sparse.FloatTensor:
        """Channel-wise sparse normalization with per-channel sparsity tracking."""
        self.mode_usage["channel_wise"] += 1
        
        # Update channel sparsity statistics
        self._update_channel_sparsities(x)
        
        # Use sparse-aware mode but with channel-specific adaptations
        return self._sparse_aware_mode(x)
    
    def _adaptive_mode(self, x: torch.sparse.FloatTensor, sparsity: float) -> torch.Tensor:
        """Adaptively choose the best normalization strategy."""
        self.mode_usage["adaptive"] += 1
        
        if sparsity > 0.9:
            # Very sparse - use sparse-aware mode
            return self._sparse_aware_mode(x)
        elif sparsity > 0.5:
            # Moderately sparse - use channel-wise mode
            return self._channel_wise_mode(x)
        else:
            # Not very sparse - use dense mode
            return self._dense_mode(x)
    
    def _update_channel_sparsities(self, x: torch.sparse.FloatTensor):
        """Update per-channel sparsity statistics."""
        self.channel_update_count += 1
        
        indices = x._indices()
        if len(indices) >= 2:
            channel_indices = indices[1]
            
            for c in range(self.num_features):
                channel_mask = channel_indices == c
                channel_nnz = channel_mask.sum().item()
                
                # Estimate channel sparsity
                total_channel_elements = x.numel() // self.num_features
                channel_sparsity = 1.0 - (channel_nnz / max(total_channel_elements, 1))
                
                # Exponential moving average
                alpha = 0.1
                self.channel_sparsities[c] = (
                    alpha * channel_sparsity + (1 - alpha) * self.channel_sparsities[c]
                )
    
    def get_normalization_stats(self) -> Dict[str, Any]:
        """Get normalization performance statistics."""
        total_ops = sum(self.mode_usage.values())
        
        return {
            "mode_usage_ratios": {
                mode: count / max(total_ops, 1) 
                for mode, count in self.mode_usage.items()
            },
            "total_operations": total_ops,
            "current_mode": self.sparse_mode,
            "channel_sparsities": self.channel_sparsities.tolist(),
            "average_channel_sparsity": self.channel_sparsities.mean().item(),
            "channel_updates": self.channel_update_count,
        }
    
    def optimize_mode(self):
        """Optimize the sparse mode based on usage patterns."""
        stats = self.get_normalization_stats()
        ratios = stats["mode_usage_ratios"]
        
        # Choose the most frequently used mode
        best_mode = max(ratios.keys(), key=lambda k: ratios[k])
        
        if best_mode != "adaptive" and ratios[best_mode] > 0.8:
            self.sparse_mode = best_mode


class AdvancedSparseActivation(nn.Module):
    """
    Advanced sparse activation with multiple activation functions and optimization strategies.
    """
    
    def __init__(self, activation: str = "relu", sparse_threshold: float = 0.1,
                 preserve_sparsity: bool = True, adaptive_threshold: bool = True,
                 activation_params: Optional[Dict[str, Any]] = None):
        super().__init__()
        
        self.activation_name = activation
        self.sparse_threshold = sparse_threshold
        self.preserve_sparsity = preserve_sparsity
        self.adaptive_threshold = adaptive_threshold
        self.activation_params = activation_params or {}
        
        # Initialize activation function
        self.activation = self._create_activation(activation, self.activation_params)
        
        # Performance tracking
        self.sparse_activations = 0
        self.dense_activations = 0
        self.sparsity_history = []
        self.threshold_history = []
    
    def _create_activation(self, activation: str, params: Dict[str, Any]) -> nn.Module:
        """Create activation function with parameters."""
        if activation == "relu":
            return nn.ReLU(inplace=params.get("inplace", False))
        elif activation == "leaky_relu":
            return nn.LeakyReLU(
                negative_slope=params.get("negative_slope", 0.01),
                inplace=params.get("inplace", False)
            )
        elif activation == "elu":
            return nn.ELU(
                alpha=params.get("alpha", 1.0),
                inplace=params.get("inplace", False)
            )
        elif activation == "selu":
            return nn.SELU(inplace=params.get("inplace", False))
        elif activation == "gelu":
            return nn.GELU()
        elif activation == "swish" or activation == "silu":
            return nn.SiLU(inplace=params.get("inplace", False))
        elif activation == "mish":
            return nn.Mish(inplace=params.get("inplace", False))
        elif activation == "tanh":
            return nn.Tanh()
        elif activation == "sigmoid":
            return nn.Sigmoid()
        elif activation == "softplus":
            return nn.Softplus(
                beta=params.get("beta", 1),
                threshold=params.get("threshold", 20)
            )
        elif activation == "prelu":
            return nn.PReLU(
                num_parameters=params.get("num_parameters", 1),
                init=params.get("init", 0.25)
            )
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_sparse and self.preserve_sparsity:
            return self._sparse_activation(x)
        else:
            return self._dense_activation(x)
    
    def _sparse_activation(self, x: torch.sparse.FloatTensor) -> torch.sparse.FloatTensor:
        """Apply activation to sparse tensor while preserving sparsity."""
        self.sparse_activations += 1
        
        indices = x._indices()
        values = x._values()
        shape = x.shape
        
        # Apply activation to non-zero values only
        activated_values = self.activation(values)
        
        # Handle activations that might create zeros (like ReLU with negative inputs)
        if self.activation_name in ["relu", "elu", "selu"]:
            # Remove values that became zero after activation
            nonzero_mask = activated_values != 0
            if nonzero_mask.any():
                filtered_indices = indices[:, nonzero_mask]
                filtered_values = activated_values[nonzero_mask]
                result = torch.sparse.FloatTensor(filtered_indices, filtered_values, shape)
            else:
                # All values became zero
                result = torch.sparse.FloatTensor(
                    torch.zeros((len(shape), 0), dtype=torch.long, device=x.device),
                    torch.zeros(0, device=x.device),
                    shape
                )
        else:
            result = torch.sparse.FloatTensor(indices, activated_values, shape)
        
        # Track sparsity
        current_sparsity = 1.0 - (result._nnz() / result.numel())
        self.sparsity_history.append(current_sparsity)
        
        # Adaptive threshold adjustment
        if self.adaptive_threshold and len(self.sparsity_history) > 10:
            self._adjust_threshold()
        
        return result
    
    def _dense_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply activation to dense tensor with optional sparsification."""
        self.dense_activations += 1
        
        out = self.activation(x)
        
        # Convert to sparse if beneficial
        if self.preserve_sparsity:
            sparsity = (out == 0).float().mean().item()
            self.sparsity_history.append(sparsity)
            
            if sparsity > self.sparse_threshold:
                return out.to_sparse()
        
        return out
    
    def _adjust_threshold(self):
        """Adjust sparse threshold based on observed sparsity patterns."""
        recent_sparsities = self.sparsity_history[-10:]
        avg_sparsity = sum(recent_sparsities) / len(recent_sparsities)
        
        # Adjust threshold to maintain good sparsity utilization
        if avg_sparsity > self.sparse_threshold + 0.2:
            # Sparsity is much higher than threshold - lower threshold
            new_threshold = max(0.05, self.sparse_threshold * 0.9)
        elif avg_sparsity < self.sparse_threshold - 0.1:
            # Sparsity is lower than threshold - raise threshold
            new_threshold = min(0.8, self.sparse_threshold * 1.1)
        else:
            new_threshold = self.sparse_threshold
        
        if new_threshold != self.sparse_threshold:
            self.sparse_threshold = new_threshold
            self.threshold_history.append(new_threshold)
    
    def get_activation_stats(self) -> Dict[str, Any]:
        """Get activation performance statistics."""
        total_activations = self.sparse_activations + self.dense_activations
        
        return {
            "activation_type": self.activation_name,
            "sparse_activation_ratio": self.sparse_activations / max(total_activations, 1),
            "dense_activation_ratio": self.dense_activations / max(total_activations, 1),
            "total_activations": total_activations,
            "current_threshold": self.sparse_threshold,
            "average_sparsity": sum(self.sparsity_history) / len(self.sparsity_history) if self.sparsity_history else 0,
            "threshold_adjustments": len(self.threshold_history),
            "preserve_sparsity": self.preserve_sparsity,
            "adaptive_threshold": self.adaptive_threshold,
        }


class SparseGroupNorm3d(nn.Module):
    """
    Group normalization for sparse 3D tensors.
    
    Group normalization can be more stable than batch normalization
    for sparse tensors as it doesn't depend on batch statistics.
    """
    
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5,
                 affine: bool = True, sparse_mode: str = "adaptive"):
        super().__init__()
        
        if num_channels % num_groups != 0:
            raise ValueError("num_channels must be divisible by num_groups")
        
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.sparse_mode = sparse_mode
        
        # Standard group norm for fallback
        self.gn = nn.GroupNorm(num_groups, num_channels, eps, affine)
        
        # Group-wise sparsity tracking
        self.group_sparsities = torch.zeros(num_groups)
        self.group_update_count = 0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_sparse:
            return self.gn(x)
        
        if self.sparse_mode == "dense":
            return self._dense_group_norm(x)
        elif self.sparse_mode == "sparse":
            return self._sparse_group_norm(x)
        else:  # adaptive
            sparsity = 1.0 - (x._nnz() / x.numel())
            if sparsity > 0.7:
                return self._sparse_group_norm(x)
            else:
                return self._dense_group_norm(x)
    
    def _dense_group_norm(self, x: torch.sparse.FloatTensor) -> torch.Tensor:
        """Convert to dense, apply group norm, optionally convert back."""
        x_dense = x.to_dense()
        out = self.gn(x_dense)
        
        # Convert back to sparse if beneficial
        sparsity = (out == 0).float().mean().item()
        if sparsity > 0.3:
            return out.to_sparse()
        
        return out
    
    def _sparse_group_norm(self, x: torch.sparse.FloatTensor) -> torch.sparse.FloatTensor:
        """Apply group normalization to sparse tensor."""
        indices = x._indices()
        values = x._values()
        shape = x.shape
        
        if len(indices) < 2:
            # Fallback to dense mode
            return self._dense_group_norm(x)
        
        channel_indices = indices[1]
        channels_per_group = self.num_channels // self.num_groups
        
        normalized_values = torch.zeros_like(values)
        
        for g in range(self.num_groups):
            # Get channels for this group
            group_start = g * channels_per_group
            group_end = (g + 1) * channels_per_group
            
            group_mask = (channel_indices >= group_start) & (channel_indices < group_end)
            
            if not group_mask.any():
                continue
            
            group_values = values[group_mask]
            
            # Normalize group values
            mean = group_values.mean()
            var = group_values.var(unbiased=False)
            normalized_group_values = (group_values - mean) / torch.sqrt(var + self.eps)
            
            # Apply affine transformation if enabled
            if self.affine:
                # Get the corresponding channels for affine parameters
                group_channels = channel_indices[group_mask]
                for i, channel in enumerate(torch.unique(group_channels)):
                    channel_mask = group_channels == channel
                    normalized_group_values[channel_mask] = (
                        normalized_group_values[channel_mask] * self.gn.weight[channel] + 
                        self.gn.bias[channel]
                    )
            
            normalized_values[group_mask] = normalized_group_values
        
        return torch.sparse.FloatTensor(indices, normalized_values, shape)
    
    def get_group_stats(self) -> Dict[str, Any]:
        """Get group normalization statistics."""
        return {
            "num_groups": self.num_groups,
            "num_channels": self.num_channels,
            "group_sparsities": self.group_sparsities.tolist(),
            "sparse_mode": self.sparse_mode,
            "group_updates": self.group_update_count,
        }


class SparseInstanceNorm3d(nn.Module):
    """
    Instance normalization for sparse 3D tensors.
    
    Instance normalization normalizes each sample independently,
    which can be beneficial for sparse tensors.
    """
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = False, track_running_stats: bool = False,
                 sparse_mode: str = "adaptive"):
        super().__init__()
        
        self.num_features = num_features
        self.eps = eps
        self.sparse_mode = sparse_mode
        
        # Standard instance norm
        self.instance_norm = nn.InstanceNorm3d(
            num_features, eps, momentum, affine, track_running_stats
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_sparse:
            return self.instance_norm(x)
        
        if self.sparse_mode == "dense":
            return self._dense_instance_norm(x)
        elif self.sparse_mode == "sparse":
            return self._sparse_instance_norm(x)
        else:  # adaptive
            sparsity = 1.0 - (x._nnz() / x.numel())
            if sparsity > 0.8:
                return self._sparse_instance_norm(x)
            else:
                return self._dense_instance_norm(x)
    
    def _dense_instance_norm(self, x: torch.sparse.FloatTensor) -> torch.Tensor:
        """Convert to dense, apply instance norm, optionally convert back."""
        x_dense = x.to_dense()
        out = self.instance_norm(x_dense)
        
        sparsity = (out == 0).float().mean().item()
        if sparsity > 0.3:
            return out.to_sparse()
        
        return out
    
    def _sparse_instance_norm(self, x: torch.sparse.FloatTensor) -> torch.sparse.FloatTensor:
        """Apply instance normalization to sparse tensor."""
        indices = x._indices()
        values = x._values()
        shape = x.shape
        
        if len(indices) < 2:
            return self._dense_instance_norm(x)
        
        batch_indices = indices[0]
        channel_indices = indices[1]
        
        normalized_values = torch.zeros_like(values)
        
        # Normalize per instance and channel
        for b in range(shape[0]):  # batch
            for c in range(shape[1]):  # channel
                mask = (batch_indices == b) & (channel_indices == c)
                
                if not mask.any():
                    continue
                
                instance_values = values[mask]
                
                # Normalize
                mean = instance_values.mean()
                var = instance_values.var(unbiased=False)
                normalized_instance_values = (instance_values - mean) / torch.sqrt(var + self.eps)
                
                # Apply affine transformation if enabled
                if self.instance_norm.affine:
                    normalized_instance_values = (
                        normalized_instance_values * self.instance_norm.weight[c] + 
                        self.instance_norm.bias[c]
                    )
                
                normalized_values[mask] = normalized_instance_values
        
        return torch.sparse.FloatTensor(indices, normalized_values, shape)


# ============================================================================
# SPARSE NORMALIZATION AND ACTIVATION UTILITIES
# ============================================================================

def create_sparse_normalization_layer(norm_type: str, num_features: int, **kwargs) -> nn.Module:
    """
    Factory function to create sparse normalization layers.
    
    Args:
        norm_type: Type of normalization ("batch", "group", "instance", "layer")
        num_features: Number of features/channels
        **kwargs: Layer-specific arguments
        
    Returns:
        Sparse normalization layer
    """
    if norm_type == "batch":
        return AdvancedSparseBatchNorm3d(num_features, **kwargs)
    elif norm_type == "group":
        num_groups = kwargs.pop("num_groups", min(32, num_features))
        return SparseGroupNorm3d(num_groups, num_features, **kwargs)
    elif norm_type == "instance":
        return SparseInstanceNorm3d(num_features, **kwargs)
    elif norm_type == "layer":
        return SparseLayerNorm3d([num_features], **kwargs)
    else:
        raise ValueError(f"Unknown normalization type: {norm_type}")


def create_sparse_activation_layer(activation: str, **kwargs) -> nn.Module:
    """
    Factory function to create sparse activation layers.
    
    Args:
        activation: Type of activation
        **kwargs: Activation-specific arguments
        
    Returns:
        Sparse activation layer
    """
    return AdvancedSparseActivation(activation, **kwargs)


def optimize_normalization_for_sparsity(model: nn.Module, 
                                       sparse_threshold: float = 0.1) -> nn.Module:
    """
    Replace standard normalization layers with sparse-aware versions.
    
    Args:
        model: Model to optimize
        sparse_threshold: Sparsity threshold for conversions
        
    Returns:
        Optimized model
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm3d) and not isinstance(module, (SparseBatchNorm3d, AdvancedSparseBatchNorm3d)):
            # Replace with advanced sparse batch norm
            sparse_bn = AdvancedSparseBatchNorm3d(
                module.num_features, module.eps, module.momentum,
                module.affine, module.track_running_stats,
                sparsity_threshold=sparse_threshold
            )
            
            # Copy parameters
            if module.affine:
                sparse_bn.bn.weight.data = module.weight.data
                sparse_bn.bn.bias.data = module.bias.data
                sparse_bn.sparse_weight.data = module.weight.data
                sparse_bn.sparse_bias.data = module.bias.data
            
            if module.track_running_stats:
                sparse_bn.bn.running_mean.data = module.running_mean.data
                sparse_bn.bn.running_var.data = module.running_var.data
                sparse_bn.sparse_running_mean.data = module.running_mean.data
                sparse_bn.sparse_running_var.data = module.running_var.data
            
            # Replace in model
            _replace_module_in_model(model, name, sparse_bn)
        
        elif isinstance(module, nn.GroupNorm):
            # Replace with sparse group norm
            sparse_gn = SparseGroupNorm3d(
                module.num_groups, module.num_channels, module.eps, module.affine
            )
            
            # Copy parameters
            if module.affine:
                sparse_gn.gn.weight.data = module.weight.data
                sparse_gn.gn.bias.data = module.bias.data
            
            # Replace in model
            _replace_module_in_model(model, name, sparse_gn)
    
    return model


def analyze_normalization_sparsity(model: nn.Module, sample_input: torch.Tensor) -> Dict[str, Any]:
    """
    Analyze sparsity patterns in normalization layers.
    
    Args:
        model: Model to analyze
        sample_input: Sample input tensor
        
    Returns:
        Normalization sparsity analysis
    """
    model.eval()
    normalization_info = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(module, (nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d)):
                input_tensor = input[0] if isinstance(input, tuple) else input
                output_tensor = output
                
                # Calculate sparsity before and after normalization
                if input_tensor.is_sparse:
                    input_sparsity = 1.0 - (input_tensor._nnz() / input_tensor.numel())
                else:
                    input_sparsity = (input_tensor == 0).float().mean().item()
                
                if output_tensor.is_sparse:
                    output_sparsity = 1.0 - (output_tensor._nnz() / output_tensor.numel())
                else:
                    output_sparsity = (output_tensor == 0).float().mean().item()
                
                normalization_info[name] = {
                    "layer_type": type(module).__name__,
                    "input_sparsity": input_sparsity,
                    "output_sparsity": output_sparsity,
                    "sparsity_change": output_sparsity - input_sparsity,
                    "input_is_sparse": input_tensor.is_sparse,
                    "output_is_sparse": output_tensor.is_sparse,
                }
        return hook
    
    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d)):
            hook = module.register_forward_hook(hook_fn(name))
            hooks.append(hook)
    
    # Forward pass
    with torch.no_grad():
        _ = model(sample_input)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Calculate summary statistics
    if normalization_info:
        sparsity_changes = [info["sparsity_change"] for info in normalization_info.values()]
        
        summary = {
            "layer_analysis": normalization_info,
            "average_sparsity_change": sum(sparsity_changes) / len(sparsity_changes),
            "max_sparsity_increase": max(sparsity_changes),
            "max_sparsity_decrease": min(sparsity_changes),
            "num_layers_analyzed": len(normalization_info),
        }
    else:
        summary = {
            "layer_analysis": {},
            "num_layers_analyzed": 0,
        }
    
    return summary