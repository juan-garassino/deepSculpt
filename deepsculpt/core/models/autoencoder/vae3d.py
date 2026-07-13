"""3D KL-regularized autoencoder for latent diffusion (Stage B).

Compresses 64³×1 binary occupancy volumes to a 16³×latent_channels field the
diffusion UNet trains on (~64× fewer voxels per step). Fully convolutional on
purpose — no flattened Linear anywhere — so a future 32³-latent / 128³-volume
run needs no architecture surgery (the Stage-C constraint).

Conventions:
- input volumes are [0, 1] occupancy (threshold-0.5 world);
- logits stay internal: ``decode()`` returns sigmoid probabilities unless the
  trainer explicitly asks for logits for its BCE;
- GroupNorm, not BatchNorm — the VAE runs frozen-eval inside bf16 diffusion
  training, where BN running-stats semantics are a known footgun.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class _ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class VAE3D(nn.Module):
    """KL-VAE: 64³×in_channels -> (down ×2) -> 16³×latent_channels -> back."""

    def __init__(
        self,
        in_channels: int = 1,
        latent_channels: int = 4,
        base_channels: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.base_channels = base_channels
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4

        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, c1, 3, padding=1),
            _ResBlock(c1),
            nn.Conv3d(c1, c2, 3, stride=2, padding=1),   # 64 -> 32
            _ResBlock(c2),
            nn.Conv3d(c2, c3, 3, stride=2, padding=1),   # 32 -> 16
            _ResBlock(c3),
            nn.GroupNorm(8, c3),
            nn.SiLU(),
            nn.Conv3d(c3, 2 * latent_channels, 1),        # -> mu, logvar
        )

        self.decoder = nn.Sequential(
            nn.Conv3d(latent_channels, c3, 3, padding=1),
            _ResBlock(c3),
            nn.Upsample(scale_factor=2, mode="nearest"),  # 16 -> 32
            nn.Conv3d(c3, c2, 3, padding=1),
            _ResBlock(c2),
            nn.Upsample(scale_factor=2, mode="nearest"),  # 32 -> 64
            nn.Conv3d(c2, c1, 3, padding=1),
            _ResBlock(c1),
            nn.GroupNorm(8, c1),
            nn.SiLU(),
            nn.Conv3d(c1, in_channels, 3, padding=1),     # -> logits
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """[B, C, D, H, W] in [0,1] -> (mu, logvar), each [B, latent_ch, D/4, ...]."""
        mu, logvar = self.encoder(x).chunk(2, dim=1)
        return mu, torch.clamp(logvar, -30.0, 20.0)

    def decode(self, z: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        logits = self.decoder(z)
        return logits if return_logits else torch.sigmoid(logits)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Training forward: (recon_logits, mu, logvar) with reparameterized z."""
        mu, logvar = self.encode(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.decode(z, return_logits=True), mu, logvar

    def get_model_info(self) -> dict:
        return {
            "model_type": "vae3d",
            "in_channels": self.in_channels,
            "latent_channels": self.latent_channels,
            "base_channels": self.base_channels,
            "parameters": sum(p.numel() for p in self.parameters()),
        }
