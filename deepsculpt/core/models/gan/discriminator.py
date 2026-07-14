"""
GAN Discriminator models for DeepSculpt PyTorch implementation.

This module contains all discriminator architectures for 3D GAN models,
including simple, complex, progressive, and conditional discriminators.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List, Union
import math

from ..base_models import SEMANTIC_CLASSES, BaseDiscriminator, SparseConv3d, SparseBatchNorm3d


class SimpleDiscriminator(BaseDiscriminator):
    """Simple discriminator model equivalent to TensorFlow version."""
    
    def __init__(self, void_dim: int = 64, color_mode: int = 1, sparse: bool = False):
        super().__init__(void_dim, color_mode, sparse)
        
        # Convolution layers
        Conv = SparseConv3d if sparse else nn.Conv3d
        BatchNorm = SparseBatchNorm3d if sparse else nn.BatchNorm3d
        
        self.conv1 = Conv(self.input_channels, 64, 4, 2, 1, bias=False)
        self.bn1 = BatchNorm(64)
        
        self.conv2 = Conv(64, 128, 4, 2, 1, bias=False)
        self.bn2 = BatchNorm(128)
        
        self.conv3 = Conv(128, 256, 4, 2, 1, bias=False)
        self.bn3 = BatchNorm(256)
        
        self.conv4 = Conv(256, 512, 4, 2, 1, bias=False)
        self.bn4 = BatchNorm(512)
        
        # Final classification layer
        final_size = void_dim // 16  # After 4 conv layers with stride 2
        self.fc = nn.Linear(512 * final_size ** 3, 1)
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input is already in PyTorch channels-first format: (batch, channels, depth, height, width)
        
        # Convolution blocks
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.leaky_relu(x)
        
        # Flatten and classify
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        # Don't apply sigmoid - loss function expects logits
        
        return x


class ComplexDiscriminator(BaseDiscriminator):
    """Complex discriminator with additional layers and features."""
    
    def __init__(self, void_dim: int = 64, color_mode: int = 1, sparse: bool = False):
        super().__init__(void_dim, color_mode, sparse)
        
        # Convolution layers with more complexity
        Conv = SparseConv3d if sparse else nn.Conv3d
        BatchNorm = SparseBatchNorm3d if sparse else nn.BatchNorm3d
        
        self.conv1 = Conv(self.input_channels, 32, 4, 2, 1, bias=False)
        self.bn1 = BatchNorm(32)
        
        self.conv2 = Conv(32, 64, 4, 2, 1, bias=False)
        self.bn2 = BatchNorm(64)
        
        self.conv3 = Conv(64, 128, 4, 2, 1, bias=False)
        self.bn3 = BatchNorm(128)
        
        self.conv4 = Conv(128, 256, 4, 2, 1, bias=False)
        self.bn4 = BatchNorm(256)
        
        self.conv5 = Conv(256, 512, 4, 2, 1, bias=False)
        self.bn5 = BatchNorm(512)
        
        # Additional feature extraction
        self.conv6 = Conv(512, 1024, 3, 1, 1, bias=False)
        self.bn6 = BatchNorm(1024)
        
        # Final classification layers
        final_size = void_dim // 32  # After 5 conv layers with stride 2
        self.fc1 = nn.Linear(1024 * final_size ** 3, 512)
        self.fc2 = nn.Linear(512, 1)
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input is already in PyTorch channels-first format: (batch, channels, depth, height, width)
        
        # Convolution blocks
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.leaky_relu(x)
        
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.leaky_relu(x)
        
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.leaky_relu(x)
        
        # Flatten and classify
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        # Don't apply sigmoid - loss function expects logits
        
        return x


class ProgressiveDiscriminator(BaseDiscriminator):
    """Progressive discriminator for high-resolution 3D data."""
    
    def __init__(self, void_dim: int = 64, color_mode: int = 1, max_resolution: int = 128, sparse: bool = False):
        super().__init__(void_dim, color_mode, sparse)
        
        self.max_resolution = max_resolution
        
        # Progressive blocks for different resolutions
        self.progressive_blocks = nn.ModuleList()
        self.from_rgb_layers = nn.ModuleList()
        
        # Create progressive blocks for each resolution level
        current_res = max_resolution
        current_channels = 16
        
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
        
        # Final classification block
        self.final_block = self._make_final_block(current_channels)
        
        self.current_level = 0  # Current progressive level
        self.alpha = 1.0  # Blending factor for progressive growing
    
    def _make_progressive_block(self, in_channels: int, out_channels: int):
        """Create a progressive block that halves the resolution."""
        Conv = SparseConv3d if self.sparse else nn.Conv3d
        BatchNorm = SparseBatchNorm3d if self.sparse else nn.BatchNorm3d
        
        return nn.Sequential(
            Conv(in_channels, in_channels, 3, 1, 1),
            BatchNorm(in_channels),
            nn.LeakyReLU(0.2),
            Conv(in_channels, out_channels, 3, 1, 1),
            BatchNorm(out_channels),
            nn.LeakyReLU(0.2),
            nn.AvgPool3d(2, 2)
        )
    
    def _make_final_block(self, in_channels: int):
        """Create the final classification block."""
        Conv = SparseConv3d if self.sparse else nn.Conv3d
        BatchNorm = SparseBatchNorm3d if self.sparse else nn.BatchNorm3d
        
        return nn.Sequential(
            Conv(in_channels, in_channels, 3, 1, 1),
            BatchNorm(in_channels),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input is already in PyTorch channels-first format: (batch, channels, depth, height, width)
        
        # Convert from RGB at current resolution
        x = self.from_rgb_layers[self.current_level](x)
        
        # Progressive blocks up to current level
        for i in range(self.current_level, len(self.progressive_blocks)):
            x = self.progressive_blocks[i](x)
        
        # Final classification
        x = self.final_block(x)
        
        return x
    
    def grow(self):
        """Grow the network by one level."""
        if self.current_level < len(self.progressive_blocks) - 1:
            self.current_level += 1
            self.alpha = 0.0  # Start with full blend to new layer
    
    def set_alpha(self, alpha: float):
        """Set the blending factor for progressive growing."""
        self.alpha = max(0.0, min(1.0, alpha))


class ConditionalDiscriminator(BaseDiscriminator):
    """Conditional discriminator for controlled discrimination."""
    
    def __init__(self, void_dim: int = 64, color_mode: int = 1, condition_dim: int = 10, sparse: bool = False):
        super().__init__(void_dim, color_mode, sparse)
        
        self.condition_dim = condition_dim
        
        # Condition embedding
        self.condition_embedding = nn.Embedding(condition_dim, 128)
        
        # Convolution layers
        Conv = SparseConv3d if sparse else nn.Conv3d
        BatchNorm = SparseBatchNorm3d if sparse else nn.BatchNorm3d
        
        self.conv1 = Conv(self.input_channels, 64, 4, 2, 1, bias=False)
        self.bn1 = BatchNorm(64)
        
        self.conv2 = Conv(64, 128, 4, 2, 1, bias=False)
        self.bn2 = BatchNorm(128)
        
        self.conv3 = Conv(128, 256, 4, 2, 1, bias=False)
        self.bn3 = BatchNorm(256)
        
        self.conv4 = Conv(256, 512, 4, 2, 1, bias=False)
        self.bn4 = BatchNorm(512)
        
        # Final classification layer with condition
        final_size = void_dim // 16  # After 4 conv layers with stride 2
        self.fc = nn.Linear(512 * final_size ** 3 + 128, 1)  # +128 for condition embedding
        
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Input is already in PyTorch channels-first format: (batch, channels, depth, height, width)
        
        # Convolution blocks
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.leaky_relu(x)
        
        # Flatten features
        x = x.reshape(x.size(0), -1)
        
        # Add condition if provided
        if condition is not None:
            condition_embedded = self.condition_embedding(condition)
            x = torch.cat([x, condition_embedded], dim=1)
        
        # Final classification — return logits for softplus loss
        x = self.fc(x)

        return x


class SpectralNormDiscriminator(BaseDiscriminator):
    """Discriminator with spectral normalization for training stability."""
    
    def __init__(self, void_dim: int = 64, color_mode: int = 1, sparse: bool = False):
        super().__init__(void_dim, color_mode, sparse)

        if color_mode == 1:
            # Semantic-class color mode: 13 one-hot channels (empty + 12
            # shodhan element classes) — mirrors SkipGenerator's color head.
            self.input_channels = SEMANTIC_CLASSES
        elif color_mode == 2:
            # RGBA continuous field: [alpha, R, G, B].
            self.input_channels = 4

        # Convolution layers with spectral normalization
        Conv = SparseConv3d if sparse else nn.Conv3d

        self.conv1 = nn.utils.spectral_norm(Conv(self.input_channels, 64, 4, 2, 1, bias=False))
        self.conv2 = nn.utils.spectral_norm(Conv(64, 128, 4, 2, 1, bias=False))
        self.conv3 = nn.utils.spectral_norm(Conv(128, 256, 4, 2, 1, bias=False))
        self.conv4 = nn.utils.spectral_norm(Conv(256, 512, 4, 2, 1, bias=False))
        
        # Final classification layer with spectral normalization
        final_size = void_dim // 16  # After 4 conv layers with stride 2
        self.fc = nn.utils.spectral_norm(nn.Linear(512 * final_size ** 3, 1))
        
        self.leaky_relu = nn.LeakyReLU(0.2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input is already in PyTorch channels-first format: (batch, channels, depth, height, width)
        
        # Convolution blocks without batch normalization (spectral norm instead)
        x = self.conv1(x)
        x = self.leaky_relu(x)
        
        x = self.conv2(x)
        x = self.leaky_relu(x)
        
        x = self.conv3(x)
        x = self.leaky_relu(x)
        
        x = self.conv4(x)
        x = self.leaky_relu(x)
        
        # Flatten and classify
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        
        return x


class MultiScaleDiscriminator(BaseDiscriminator):
    """Multi-scale discriminator for improved training dynamics."""
    
    def __init__(self, void_dim: int = 64, color_mode: int = 1, num_scales: int = 3, sparse: bool = False):
        super().__init__(void_dim, color_mode, sparse)
        
        self.num_scales = num_scales
        self.discriminators = nn.ModuleList()

        # Each sub-discriminator must match the resolution it actually receives
        # (full, /2, /4, ...). SimpleDiscriminator needs void_dim // 16 >= 1.
        smallest = void_dim // (2 ** (num_scales - 1))
        if smallest < 16:
            raise ValueError(
                f"MultiScaleDiscriminator with num_scales={num_scales} needs "
                f"void_dim >= {16 * 2 ** (num_scales - 1)} (got void_dim={void_dim})"
            )
        for i in range(num_scales):
            scale_discriminator = SimpleDiscriminator(void_dim // (2 ** i), color_mode, sparse)
            self.discriminators.append(scale_discriminator)

        # Downsampling layers for different scales
        self.downsample = nn.AvgPool3d(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        current_x = x

        for i, discriminator in enumerate(self.discriminators):
            output = discriminator(current_x)
            outputs.append(output)

            # Downsample for next scale (except for the last one)
            if i < self.num_scales - 1:
                current_x = self.downsample(current_x)

        # Mean of per-scale logits -> (B, 1); keeps gradients flowing to all scales
        # and satisfies the trainer contract shared by every discriminator.
        return torch.stack(outputs, dim=0).mean(dim=0)


class PatchDiscriminator(BaseDiscriminator):
    """Patch-based discriminator (PatchGAN) for local discrimination."""
    
    def __init__(self, void_dim: int = 64, color_mode: int = 1, patch_size: int = 16, sparse: bool = False):
        super().__init__(void_dim, color_mode, sparse)
        
        self.patch_size = patch_size
        
        # Convolution layers for patch discrimination
        Conv = SparseConv3d if sparse else nn.Conv3d
        BatchNorm = SparseBatchNorm3d if sparse else nn.BatchNorm3d
        
        self.conv1 = Conv(self.input_channels, 64, 4, 2, 1)
        self.conv2 = Conv(64, 128, 4, 2, 1)
        self.bn2 = BatchNorm(128)
        
        self.conv3 = Conv(128, 256, 4, 2, 1)
        self.bn3 = BatchNorm(256)
        
        self.conv4 = Conv(256, 512, 4, 1, 1)
        self.bn4 = BatchNorm(512)
        
        # Final patch classification
        self.conv5 = Conv(512, 1, 4, 1, 1)

        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input is already in PyTorch channels-first format: (batch, channels, depth, height, width)

        # Convolution blocks
        x = self.conv1(x)
        x = self.leaky_relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.leaky_relu(x)
        
        # Final patch logits (B, 1, d, h, w) -> mean-pool patches to (B, 1):
        # standard PatchGAN aggregation, keeps the trainer contract uniform.
        x = self.conv5(x)
        x = x.mean(dim=[2, 3, 4])

        return x


class LightDiscriminator(BaseDiscriminator):
    """Lightweight projection discriminator for 3D sparse voxel GANs.

    Deliberately small (~500K params) to balance against a stronger generator.
    Features:
    - Spectral norm for Lipschitz stability
    - Projection head: judges mean + variance of features (global statistics)
      instead of specific voxel patterns — harder to memorize
    - Returns intermediate features for feature matching loss
    """

    def __init__(self, void_dim: int = 64, color_mode: int = 1, sparse: bool = False):
        super().__init__(void_dim, color_mode, sparse)
        Conv = SparseConv3d if sparse else nn.Conv3d
        ch = 32
        self.conv1 = nn.utils.spectral_norm(Conv(self.input_channels, ch, 4, 2, 1))
        self.conv2 = nn.utils.spectral_norm(Conv(ch, ch * 2, 4, 2, 1))
        self.conv3 = nn.utils.spectral_norm(Conv(ch * 2, ch * 4, 4, 2, 1))
        self.pool = nn.AdaptiveAvgPool3d(1)
        # Projection: mean + variance of features → harder to overfit
        self.fc = nn.utils.spectral_norm(nn.Linear(ch * 8, 1))
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        f1 = self.lrelu(self.conv1(x))
        f2 = self.lrelu(self.conv2(f1))
        f3 = self.lrelu(self.conv3(f2))

        # Projection: global mean + spatial variance
        pooled = self.pool(f3).flatten(1)         # (B, ch*4)
        std_pool = f3.std(dim=[2, 3, 4])          # (B, ch*4)
        proj = torch.cat([pooled, std_pool], dim=1)  # (B, ch*8)
        logit = self.fc(proj)

        if return_features:
            return logit, [f1, f2, f3]
        return logit