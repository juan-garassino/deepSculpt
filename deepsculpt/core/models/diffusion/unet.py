"""
3D U-Net architecture for diffusion models in DeepSculpt.

This module implements a 3D U-Net with time embedding and conditioning
for diffusion-based 3D sculpture generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from typing import Optional, Tuple, Dict, Any, List, Union
import math

from ..base_models import BaseDiffusionModel, SparseConv3d, SparseConvTranspose3d, SparseBatchNorm3d


class TimeEmbedding(nn.Module):
    """
    Time embedding layer for diffusion timesteps.
    
    Uses sinusoidal embeddings similar to transformer positional encodings.
    """
    
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Create sinusoidal embedding weights
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        self.register_buffer('emb', emb)
        
        # Linear layers to process embeddings
        # Output dimension should match embedding_dim for compatibility with ResBlocks
        self.linear1 = nn.Linear(embedding_dim, embedding_dim * 4)
        self.linear2 = nn.Linear(embedding_dim * 4, embedding_dim)  # Output same as input
        self.act = nn.SiLU()
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        # Create sinusoidal embeddings
        emb = timesteps[:, None] * self.emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        # Process through linear layers
        emb = self.linear1(emb)
        emb = self.act(emb)
        emb = self.linear2(emb)
        
        return emb


class ResBlock3D(nn.Module):
    """
    3D Residual block with time embedding and optional conditioning.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        sparse: bool = False,
        dropout: float = 0.1
    ):
        super().__init__()
        
        Conv = SparseConv3d if sparse else nn.Conv3d
        BatchNorm = SparseBatchNorm3d if sparse else nn.BatchNorm3d
        
        self.conv1 = Conv(in_channels, out_channels, 3, 1, 1)
        self.bn1 = BatchNorm(out_channels)
        
        self.conv2 = Conv(out_channels, out_channels, 3, 1, 1)
        self.bn2 = BatchNorm(out_channels)
        
        # Time embedding projection
        self.time_proj = nn.Linear(time_emb_dim, out_channels)
        
        # Skip connection
        if in_channels != out_channels:
            self.skip = Conv(in_channels, out_channels, 1, 1, 0)
        else:
            self.skip = nn.Identity()
        
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        skip = self.skip(x)
        
        # First conv block
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.act(h)
        
        # Add time embedding
        time_proj = self.time_proj(time_emb)
        # Reshape time embedding to match spatial dimensions
        while len(time_proj.shape) < len(h.shape):
            time_proj = time_proj.unsqueeze(-1)
        h = h + time_proj
        
        # Second conv block
        h = self.dropout(h)
        h = self.conv2(h)
        h = self.bn2(h)
        
        # Add skip connection and activate
        h = h + skip
        h = self.act(h)
        
        return h


class AttentionBlock3D(nn.Module):
    """
    3D self-attention block for improved feature learning.
    """
    
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv3d(channels, channels * 3, 1)
        self.proj = nn.Conv3d(channels, channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, depth, height, width = x.shape
        
        # Normalize input
        h = self.norm(x)
        
        # Compute Q, K, V
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        
        # SDPA (flash/mem-efficient kernels): never materializes the seq×seq
        # attention matrix — the einsum version allocated B×heads×4096×4096
        # at 16³ resolution and OOM'd the 22 GiB L4 on its own.
        spatial_size = depth * height * width
        q = q.reshape(batch, self.num_heads, self.head_dim, spatial_size).transpose(-1, -2)
        k = k.reshape(batch, self.num_heads, self.head_dim, spatial_size).transpose(-1, -2)
        v = v.reshape(batch, self.num_heads, self.head_dim, spatial_size).transpose(-1, -2)

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(-1, -2).reshape(batch, channels, depth, height, width)
        
        # Project and add residual
        out = self.proj(out)
        out = out + x
        
        return out


class UNet3D(BaseDiffusionModel):
    """
    3D U-Net for diffusion models with time embedding and attention.
    """
    
    def __init__(
        self,
        void_dim: int = 64,
        in_channels: int = 6,
        out_channels: int = 6,
        timesteps: int = 1000,
        model_channels: int = 128,
        num_res_blocks: int = 2,
        attention_resolutions: List[int] = [16, 8],
        channel_mult: List[int] = [1, 2, 4, 8],
        num_heads: int = 8,
        sparse: bool = False,
        dropout: float = 0.1,
        conditioning_dim: Optional[int] = None,
        use_checkpoint: bool = False
    ):
        super().__init__(void_dim, in_channels, out_channels, timesteps, sparse)

        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.channel_mult = channel_mult
        self.num_heads = num_heads
        self.conditioning_dim = conditioning_dim
        # Gradient checkpointing: recompute block activations in backward.
        # Without it the 64ch UNet at void 64 only fits the 22GiB L4 at
        # batch 2 (~25 min/epoch); with it batch 16 fits. BN running stats
        # see each batch twice under recompute — accepted, standard caveat.
        self.use_checkpoint = use_checkpoint
        
        # Time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = TimeEmbedding(time_embed_dim)
        
        # Conditioning embedding (if provided)
        if conditioning_dim is not None:
            self.cond_embed = nn.Linear(conditioning_dim, time_embed_dim)
        
        Conv = SparseConv3d if sparse else nn.Conv3d
        ConvTranspose = SparseConvTranspose3d if sparse else nn.ConvTranspose3d
        
        # Input projection
        self.input_proj = Conv(in_channels, model_channels, 3, 1, 1)
        
        # Encoder (downsampling path)
        self.encoder_blocks = nn.ModuleList()
        self.encoder_downsample = nn.ModuleList()
        
        ch = model_channels
        input_block_channels = [ch]
        
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock3D(ch, mult * model_channels, time_embed_dim, sparse, dropout)
                ]
                ch = mult * model_channels
                
                # Add attention at specified resolutions
                current_res = void_dim // (2 ** level)
                if current_res in attention_resolutions:
                    layers.append(AttentionBlock3D(ch, num_heads))
                
                self.encoder_blocks.append(nn.Sequential(*layers))
                input_block_channels.append(ch)
            
            # Downsample (except for the last level)
            if level < len(channel_mult) - 1:
                self.encoder_downsample.append(Conv(ch, ch, 3, 2, 1))
                input_block_channels.append(ch)
            else:
                self.encoder_downsample.append(nn.Identity())
        
        # Middle block
        self.middle_block = nn.Sequential(
            ResBlock3D(ch, ch, time_embed_dim, sparse, dropout),
            AttentionBlock3D(ch, num_heads),
            ResBlock3D(ch, ch, time_embed_dim, sparse, dropout)
        )
        
        # Decoder (upsampling path)
        self.decoder_blocks = nn.ModuleList()
        self.decoder_upsample = nn.ModuleList()
        
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                # Skip connection from encoder
                skip_ch = input_block_channels.pop()
                layers = [
                    ResBlock3D(ch + skip_ch, mult * model_channels, time_embed_dim, sparse, dropout)
                ]
                ch = mult * model_channels
                
                # Add attention at specified resolutions
                current_res = void_dim // (2 ** level)
                if current_res in attention_resolutions:
                    layers.append(AttentionBlock3D(ch, num_heads))
                
                self.decoder_blocks.append(nn.Sequential(*layers))
                
                # Upsample (except for the last iteration of the last level)
                if level > 0 and i == num_res_blocks:
                    self.decoder_upsample.append(
                        ConvTranspose(ch, ch, 4, 2, 1)
                    )
                else:
                    self.decoder_upsample.append(nn.Identity())
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            Conv(ch, out_channels, 3, 1, 1)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Accept legacy channels-last input (B, D, H, W, C) and convert to
        # channels-first. Only treat it as channels-last when dim 1 does NOT
        # already match in_channels — otherwise (B, C, D, H, W) with D == C
        # would be permuted by mistake.
        channels_last_input = (
            len(x.shape) == 5
            and x.shape[1] != self.in_channels
            and x.shape[-1] == self.in_channels
        )
        if channels_last_input:
            x = x.permute(0, 4, 1, 2, 3)
        
        # Time embedding
        time_emb = self.time_embed(timestep)
        
        # Add conditioning if provided
        if conditioning is not None and self.conditioning_dim is not None:
            cond_emb = self.cond_embed(conditioning)
            time_emb = time_emb + cond_emb
        
        # Input projection
        h = self.input_proj(x)

        ckpt = self.use_checkpoint and self.training

        def _run_block(block, h, time_emb):
            def _inner(h_):
                out = block[0](h_, time_emb)
                if len(block) > 1:
                    out = block[1](out)
                return out
            if ckpt:
                return torch.utils.checkpoint.checkpoint(_inner, h, use_reentrant=False)
            return _inner(h)

        # Encoder path with skip connections
        skip_connections = [h]  # Start with input projection output

        block_idx = 0
        for level_idx in range(len(self.channel_mult)):
            # Process all res_blocks at this level
            for _ in range(self.num_res_blocks):
                if block_idx < len(self.encoder_blocks):
                    h = _run_block(self.encoder_blocks[block_idx], h, time_emb)
                    skip_connections.append(h)
                    block_idx += 1

            # Downsample after all blocks at this level (except last)
            if level_idx < len(self.encoder_downsample):
                downsample = self.encoder_downsample[level_idx]
                if not isinstance(downsample, nn.Identity):
                    h = downsample(h)
                    skip_connections.append(h)

        # Middle block
        h = self.middle_block[0](h, time_emb)  # First ResBlock
        h = self.middle_block[1](h)  # Attention
        h = self.middle_block[2](h, time_emb)  # Second ResBlock

        # Decoder path with skip connections
        for i, (block, upsample) in enumerate(zip(self.decoder_blocks, self.decoder_upsample)):
            # Add skip connection if available
            if skip_connections:
                skip = skip_connections.pop()
                h = torch.cat([h, skip], dim=1)

            h = _run_block(block, h, time_emb)

            if not isinstance(upsample, nn.Identity):
                h = upsample(h)
        
        # Output projection
        h = self.output_proj(h)
        
        # Mirror the input convention: only emit channels-last if the caller
        # passed channels-last. Channels-first in -> channels-first out.
        if channels_last_input:
            h = h.permute(0, 2, 3, 4, 1)

        return h


class ConditionalUNet3D(UNet3D):
    """
    Conditional 3D U-Net with additional conditioning mechanisms.
    """
    
    def __init__(
        self,
        void_dim: int = 64,
        in_channels: int = 6,
        out_channels: int = 6,
        timesteps: int = 1000,
        model_channels: int = 128,
        num_res_blocks: int = 2,
        attention_resolutions: List[int] = [16, 8],
        channel_mult: List[int] = [1, 2, 4, 8],
        num_heads: int = 8,
        sparse: bool = False,
        dropout: float = 0.1,
        conditioning_dim: int = 512,
        num_classes: Optional[int] = None
    ):
        super().__init__(
            void_dim, in_channels, out_channels, timesteps, model_channels,
            num_res_blocks, attention_resolutions, channel_mult, num_heads,
            sparse, dropout, conditioning_dim
        )
        
        self.num_classes = num_classes
        
        # Class embedding for discrete conditioning
        if num_classes is not None:
            self.class_embed = nn.Embedding(num_classes, conditioning_dim)
        
        # Cross-attention layers for conditioning
        self.cross_attention_layers = nn.ModuleList()
        for level, mult in enumerate(channel_mult):
            ch = mult * model_channels
            self.cross_attention_layers.append(
                CrossAttentionBlock3D(ch, conditioning_dim, num_heads)
            )
    
    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Process class labels if provided
        if class_labels is not None and self.num_classes is not None:
            class_emb = self.class_embed(class_labels)
            if conditioning is not None:
                conditioning = conditioning + class_emb
            else:
                conditioning = class_emb
        
        return super().forward(x, timestep, conditioning)


class CrossAttentionBlock3D(nn.Module):
    """
    3D cross-attention block for conditioning.
    """
    
    def __init__(self, channels: int, context_dim: int, num_heads: int = 8):
        super().__init__()
        self.channels = channels
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        self.norm = nn.GroupNorm(8, channels)
        self.norm_context = nn.LayerNorm(context_dim)
        
        self.q = nn.Conv3d(channels, channels, 1)
        self.k = nn.Linear(context_dim, channels)
        self.v = nn.Linear(context_dim, channels)
        self.proj = nn.Conv3d(channels, channels, 1)
    
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        batch, channels, depth, height, width = x.shape
        
        # Normalize inputs
        h = self.norm(x)
        context = self.norm_context(context)
        
        # Compute Q from spatial features
        q = self.q(h)
        q = q.view(batch, self.num_heads, self.head_dim, depth * height * width)
        
        # Compute K, V from context
        k = self.k(context)  # (batch, context_len, channels)
        v = self.v(context)
        
        k = k.view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute cross-attention
        scale = self.head_dim ** -0.5
        attn = torch.einsum('bhds,bhcs->bhdc', q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.einsum('bhdc,bhcs->bhds', attn, v)
        out = out.view(batch, channels, depth, height, width)
        
        # Project and add residual
        out = self.proj(out)
        out = out + x
        
        return out