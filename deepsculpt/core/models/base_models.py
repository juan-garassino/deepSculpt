"""
Base model classes and interfaces for DeepSculpt PyTorch models.

This module provides the foundational classes and interfaces that all DeepSculpt
models inherit from, ensuring consistency and providing common functionality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List, Union
from abc import ABC, abstractmethod

# Semantic-class ("color") voxel vocabulary for the GAN path: channel 0 = empty,
# channels 1-12 = shodhan element classes (core/data/generation/shodhan.py:
# COL, SLAB, SCREEN, PIPE_R/B/Y, VOL, EDGE, WALL_R/B/Y/G).
SEMANTIC_CLASSES = 13


class BaseGenerator(nn.Module, ABC):
    """
    Abstract base class for all generator models in DeepSculpt.
    
    Provides common functionality and interface that all generators must implement.
    """
    
    def __init__(
        self,
        void_dim: int = 64,
        noise_dim: int = 100,
        color_mode: int = 1,
        sparse: bool = False,
        device: str = "cuda"
    ):
        super().__init__()
        self.void_dim = void_dim
        self.noise_dim = noise_dim
        self.color_mode = color_mode
        self.sparse = sparse
        self.device = device
        
        # Output channels based on color mode
        # color_mode=0: monochrome (1 channel), color_mode=1: OHE color classes (6 channels)
        self.output_channels = 6 if color_mode == 1 else 1

        # Statistics tracking
        self.generation_count = 0
        self.total_forward_time = 0.0
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the generator.
        
        Args:
            x: Input noise tensor of shape (batch_size, noise_dim)
            
        Returns:
            Generated 3D sculpture tensor
        """
        pass
    
    def _apply_final_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the correct final activation based on output mode.

        - 6-ch OHE (color_mode=1): softmax over channel dim (pick one class)
        - 3-ch RGB: tanh (continuous color in [-1, 1])
        - 1-ch mono: sigmoid (binary occupancy)
        """
        if self.color_mode == 1 and self.output_channels >= 6:
            return F.softmax(x, dim=1)  # dim=1 = channel dim in NCDHW
        elif self.output_channels == 3:
            return torch.tanh(x)
        else:
            return torch.sigmoid(x)

    def generate_sample(self, batch_size: int = 1) -> torch.Tensor:
        """
        Generate a sample using random noise.
        
        Args:
            batch_size: Number of samples to generate
            
        Returns:
            Generated samples tensor
        """
        noise = torch.randn(batch_size, self.noise_dim, device=self.device)
        return self.forward(noise)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "model_type": self.__class__.__name__,
            "void_dim": self.void_dim,
            "noise_dim": self.noise_dim,
            "color_mode": self.color_mode,
            "sparse": self.sparse,
            "output_channels": self.output_channels,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "generation_count": self.generation_count,
            "avg_forward_time": self.total_forward_time / max(self.generation_count, 1),
        }
    
    def reset_stats(self):
        """Reset generation statistics."""
        self.generation_count = 0
        self.total_forward_time = 0.0


class BaseDiscriminator(nn.Module, ABC):
    """
    Abstract base class for all discriminator models in DeepSculpt.
    
    Provides common functionality and interface that all discriminators must implement.
    """
    
    def __init__(
        self,
        void_dim: int = 64,
        color_mode: int = 1,
        sparse: bool = False,
        device: str = "cuda"
    ):
        super().__init__()
        self.void_dim = void_dim
        self.color_mode = color_mode
        self.sparse = sparse
        self.device = device
        
        # Input channels based on color mode
        # color_mode=0: monochrome (1 channel), color_mode=1: color (6 channels for RGB)
        self.input_channels = 6 if color_mode == 1 else 1
        
        # Statistics tracking
        self.discrimination_count = 0
        self.total_forward_time = 0.0
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the discriminator.
        
        Args:
            x: Input 3D sculpture tensor
            
        Returns:
            Discrimination score (real/fake probability)
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "model_type": self.__class__.__name__,
            "void_dim": self.void_dim,
            "color_mode": self.color_mode,
            "sparse": self.sparse,
            "input_channels": self.input_channels,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "discrimination_count": self.discrimination_count,
            "avg_forward_time": self.total_forward_time / max(self.discrimination_count, 1),
        }
    
    def reset_stats(self):
        """Reset discrimination statistics."""
        self.discrimination_count = 0
        self.total_forward_time = 0.0


class BaseDiffusionModel(nn.Module, ABC):
    """
    Abstract base class for diffusion models in DeepSculpt.
    
    Provides common functionality for 3D diffusion models.
    """
    
    def __init__(
        self,
        void_dim: int = 64,
        in_channels: int = 6,
        out_channels: int = 6,
        timesteps: int = 1000,
        sparse: bool = False,
        device: str = "cuda"
    ):
        super().__init__()
        self.void_dim = void_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.timesteps = timesteps
        self.sparse = sparse
        self.device = device
        
        # Statistics tracking
        self.denoising_count = 0
        self.total_forward_time = 0.0
    
    @abstractmethod
    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the diffusion model.
        
        Args:
            x: Noisy input tensor
            timestep: Current timestep in diffusion process
            
        Returns:
            Denoised output tensor
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "model_type": self.__class__.__name__,
            "void_dim": self.void_dim,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "timesteps": self.timesteps,
            "sparse": self.sparse,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "denoising_count": self.denoising_count,
            "avg_forward_time": self.total_forward_time / max(self.denoising_count, 1),
        }
    
    def reset_stats(self):
        """Reset denoising statistics."""
        self.denoising_count = 0
        self.total_forward_time = 0.0


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


class SparseConvTranspose3d(nn.Module):
    """
    3D transposed convolution for sparse tensors with adaptive sparsity management.
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, output_padding: int = 0, bias: bool = True,
                 sparse_threshold: float = 0.1, auto_sparse: bool = True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=bias
        )
        self.sparse_threshold = sparse_threshold
        self.auto_sparse = auto_sparse
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_sparse:
            x_dense = x.to_dense()
            out = self.conv_transpose(x_dense)
            
            if self.auto_sparse:
                sparsity = (out == 0).float().mean().item()
                if sparsity > self.sparse_threshold:
                    return out.to_sparse()
            return out
        else:
            out = self.conv_transpose(x)
            
            if self.auto_sparse:
                sparsity = (out == 0).float().mean().item()
                if sparsity > self.sparse_threshold:
                    return out.to_sparse()
            return out


class SparseBatchNorm3d(nn.Module):
    """
    Batch normalization adapted for sparse tensors.
    """
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = True, track_running_stats: bool = True):
        super().__init__()
        self.bn = nn.BatchNorm3d(num_features, eps=eps, momentum=momentum, 
                                affine=affine, track_running_stats=track_running_stats)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_sparse:
            # Convert to dense, normalize, then decide on output format
            x_dense = x.to_dense()
            out = self.bn(x_dense)
            
            # Maintain sparsity if beneficial
            sparsity = (out == 0).float().mean().item()
            if sparsity > 0.1:  # Threshold for maintaining sparsity
                return out.to_sparse()
            return out
        else:
            return self.bn(x)