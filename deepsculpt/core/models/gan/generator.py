"""
GAN Generator models for DeepSculpt PyTorch implementation.

This module contains all generator architectures for 3D GAN models,
including simple, complex, skip, monochrome, autoencoder, progressive,
and conditional generators.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List, Union
import math

from ..base_models import BaseGenerator, SparseConv3d, SparseConvTranspose3d, SparseBatchNorm3d


class SelfAttention3D(nn.Module):
    """Multi-head self-attention for 3D feature maps.

    Applied at mid-resolution in the generator so every voxel can attend
    to every other voxel — lets the network coordinate global structure
    (floors, walls, columns) instead of just local 3×3×3 patterns.
    """

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        # channels must split evenly across heads and norm groups; fall back to
        # the largest divisor so arbitrary widths (e.g. noise_dim=100 -> 50) work.
        self.num_heads = next(h for h in range(min(num_heads, channels), 0, -1) if channels % h == 0)
        self.head_dim = channels // self.num_heads
        num_groups = next(g for g in range(min(8, channels), 0, -1) if channels % g == 0)
        self.norm = nn.GroupNorm(num_groups, channels)
        self.qkv = nn.Conv3d(channels, channels * 3, 1)
        self.proj = nn.Conv3d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        S = D * H * W  # spatial tokens

        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, self.head_dim, S)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        # Scaled dot-product attention
        attn = (q.transpose(-1, -2) @ k) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        out = (v @ attn.transpose(-1, -2)).reshape(B, C, D, H, W)

        return x + self.proj(out)  # residual connection


class SimpleGenerator(BaseGenerator):
    """Simple generator model equivalent to TensorFlow version."""
    
    def __init__(self, void_dim: int = 64, noise_dim: int = 100, color_mode: int = 1, sparse: bool = False):
        super().__init__(void_dim, noise_dim, color_mode, sparse)
        
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
        x = self._apply_final_activation(x)

        return x


class ComplexGenerator(BaseGenerator):
    """Complex generator model with skip connections."""
    
    def __init__(self, void_dim: int = 64, noise_dim: int = 100, color_mode: int = 1, sparse: bool = False):
        super().__init__(void_dim, noise_dim, color_mode, sparse)
        
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

        # Final layer with skip connection — channels-first (B, C, D, H, W)
        x = torch.cat([x, skip_connections[-1]], dim=1)
        x = self.conv4(x)
        x = self._apply_final_activation(x)

        return x


class _StraightThroughBinarize(torch.autograd.Function):
    """Binarize in forward pass (round to 0/1), pass gradient straight through."""

    @staticmethod
    def forward(ctx, x):
        return (x > 0.5).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def straight_through_binarize(x: torch.Tensor) -> torch.Tensor:
    return _StraightThroughBinarize.apply(x)


class SkipGenerator(BaseGenerator):
    """Generator with skip connections and self-attention for sparse 3D voxel generation.

    Designed for the 3D sparse voxel problem where the generator needs much
    more capacity than the discriminator. Features:
    - Configurable channel width via gen_channels (default: noise_dim)
    - Self-attention at mid-resolution for global structure coordination
    - Multi-scale skip connections for gradient flow
    - STE binarization for monochrome mode
    """

    def __init__(self, void_dim: int = 64, noise_dim: int = 100, color_mode: int = 1,
                 sparse: bool = False, gen_channels: Optional[int] = None):
        super().__init__(void_dim, noise_dim, color_mode, sparse)

        ch = gen_channels or noise_dim  # backwards compatible
        self.gen_channels = ch
        self.initial_size = void_dim // 8

        ConvTranspose = SparseConvTranspose3d if sparse else nn.ConvTranspose3d
        BatchNorm = SparseBatchNorm3d if sparse else nn.BatchNorm3d

        # Dense projection from noise to spatial features
        self.fc = nn.Linear(noise_dim, self.initial_size ** 3 * ch, bias=False)
        self.bn1 = nn.BatchNorm1d(self.initial_size ** 3 * ch)

        self.conv1 = ConvTranspose(ch, ch, 3, 1, 1, bias=False)
        self.bn2 = BatchNorm(ch)

        self.conv2 = ConvTranspose(ch, ch // 2, 3, 2, 1, 1, bias=False)
        self.bn3 = BatchNorm(ch // 2)

        # Self-attention at mid-resolution (8^3 for void_dim=64, 16^3 for void_dim=128)
        self.attn = SelfAttention3D(ch // 2, num_heads=4)

        # Skip: s1 (ch channels) upsampled and concatenated
        self.conv3 = ConvTranspose(ch // 2 + ch, ch // 4, 3, 2, 1, 1, bias=False)
        self.bn4 = BatchNorm(ch // 4)

        # Skip: s2 (ch//2 channels) upsampled and concatenated
        self.conv4 = ConvTranspose(ch // 4 + ch // 2, self.output_channels, 3, 2, 1, 1, bias=True)

        # Init final bias so sigmoid(bias) ≈ 0.05 for monochrome
        if self.color_mode == 0:
            nn.init.constant_(self.conv4.bias, -2.94)

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ch = self.gen_channels

        x = self.fc(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = x.view(-1, ch, self.initial_size, self.initial_size, self.initial_size)

        # Scale 1: initial_size (e.g. 4×4×4)
        s1 = self.conv1(x)
        s1 = self.bn2(s1)
        s1 = self.relu(s1)  # (B, ch, 4, 4, 4)

        # Scale 2: 2× up (e.g. 8×8×8)
        s2 = self.conv2(s1)
        s2 = self.bn3(s2)
        s2 = self.relu(s2)  # (B, ch//2, 8, 8, 8)

        # Self-attention: every voxel attends to every other at this resolution
        s2 = self.attn(s2)

        # Scale 3: 2× up — skip from s1 (upsampled to match)
        s1_up = F.interpolate(s1, scale_factor=2, mode='trilinear', align_corners=False)
        x3 = torch.cat([s2, s1_up], dim=1)  # (B, ch//2 + ch, 8, 8, 8)
        s3 = self.conv3(x3)
        s3 = self.bn4(s3)
        s3 = self.relu(s3)  # (B, ch//4, 16, 16, 16)

        # Scale 4: 2× up — skip from s2 (upsampled to match)
        s2_up = F.interpolate(s2, scale_factor=2, mode='trilinear', align_corners=False)
        x4 = torch.cat([s3, s2_up], dim=1)  # (B, ch//4 + ch//2, 16, 16, 16)
        logits = self.conv4(x4)  # (B, output_channels, 32, 32, 32)

        if self.color_mode == 1 and self.output_channels >= 6:
            # OHE: mutually exclusive class selection
            x = F.softmax(logits, dim=1)
        elif self.output_channels == 3:
            # RGB: continuous color in [-1, 1]
            x = torch.tanh(logits)
        else:
            # Monochrome: sigmoid + straight-through binarization — output is
            # {0,1} like real data. Gradient flows via STE.
            soft = torch.sigmoid(logits)
            if self.training:
                x = straight_through_binarize(soft)
            else:
                x = (soft > 0.5).float()

        x = x.view(-1, self.output_channels, self.void_dim, self.void_dim, self.void_dim)
        return x


class MonochromeGenerator(BaseGenerator):
    """Monochrome generator model."""
    
    def __init__(self, void_dim: int = 64, noise_dim: int = 100, color_mode: int = 0, sparse: bool = False):
        # Monochrome by definition: 1 occupancy channel regardless of requested color_mode
        super().__init__(void_dim, noise_dim, 0, sparse)

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

        # Final transposed conv block — channels-first (B, C, D, H, W) like the trainer expects
        x = self.conv4(x)
        x = self._apply_final_activation(x)

        return x


class AutoencoderGenerator(BaseGenerator):
    """Generator based on autoencoder architecture."""
    
    def __init__(self, void_dim: int = 64, noise_dim: int = 100, color_mode: int = 1, sparse: bool = False):
        super().__init__(void_dim, noise_dim, color_mode, sparse)

        # Three stride-2 upsampling stages: initial_size * 8 == void_dim
        self.initial_size = void_dim // 8
        self.fc = nn.Linear(noise_dim, self.initial_size ** 3 * 16)

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
        x = x.view(-1, 16, self.initial_size, self.initial_size, self.initial_size)

        # Upsampling layers: initial_size -> *2 -> *4 -> *8 == void_dim
        x = self.conv1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self._apply_final_activation(x)

        # Channels-first (B, C, D, H, W) like the trainer expects
        return x


class ProgressiveGenerator(BaseGenerator):
    """Progressive growing generator for high-resolution 3D data."""
    
    def __init__(self, void_dim: int = 64, noise_dim: int = 100, color_mode: int = 1, 
                 max_resolution: int = 128, sparse: bool = False):
        super().__init__(void_dim, noise_dim, color_mode, sparse)
        
        self.max_resolution = max_resolution
        
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
        x = self._apply_final_activation(x)

        return x
    
    def grow(self):
        """Grow the network by one level."""
        if self.current_level < len(self.progressive_blocks) - 1:
            self.current_level += 1
            self.alpha = 0.0  # Start with full blend to new layer
    
    def set_alpha(self, alpha: float):
        """Set the blending factor for progressive growing."""
        self.alpha = max(0.0, min(1.0, alpha))


class ConditionalGenerator(BaseGenerator):
    """Conditional generator for controlled generation."""
    
    def __init__(self, void_dim: int = 64, noise_dim: int = 100, condition_dim: int = 10,
                 color_mode: int = 1, sparse: bool = False):
        super().__init__(void_dim, noise_dim, color_mode, sparse)
        
        self.condition_dim = condition_dim
        
        # Condition embedding
        self.condition_embedding = nn.Embedding(condition_dim, noise_dim)
        
        # Combined input dimension
        combined_dim = noise_dim + noise_dim  # noise + embedded condition
        
        self.initial_size = void_dim // 8
        
        # Initial dense layer with combined input
        self.fc = nn.Linear(combined_dim, self.initial_size ** 3 * noise_dim, bias=False)
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

    def forward(self, x: torch.Tensor, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        if condition is not None:
            # Embed condition
            condition_embedded = self.condition_embedding(condition)
            # Concatenate noise and condition
            x = torch.cat([x, condition_embedded], dim=1)

        # Initial dense layer and reshape
        x = self.fc(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = x.view(-1, self.noise_dim, self.initial_size, self.initial_size, self.initial_size)

        # Transposed conv blocks
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self._apply_final_activation(x)
        
        # Reshape to final output
        x = x.view(-1, self.void_dim, self.void_dim, self.void_dim, self.output_channels)
        
        return x