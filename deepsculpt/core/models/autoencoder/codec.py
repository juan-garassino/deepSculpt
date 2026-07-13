"""LatentCodec — the single place that knows how volumes map to latents.

All normalization knowledge (per-channel shift/scale) lives here and nowhere
else: the diffusion trainer encodes through it, the latent sampling pipeline
decodes through it, and the loader rebuilds it from config.json. Latents are
normalized to ~zero-mean unit-variance per channel because the samplers and
noise schedule assume symmetric data — a scalar 1/std alone would recreate
the +mean density bias this codebase already paid for at pixel level.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Tuple

import torch

from .vae3d import VAE3D


class LatentCodec:
    def __init__(
        self,
        vae: VAE3D,
        shift: torch.Tensor,   # [latent_channels]
        scale: torch.Tensor,   # [latent_channels]
        device: str = "cpu",
    ):
        self.vae = vae.to(device).eval().requires_grad_(False)
        self.device = device
        self.shift = shift.to(device).view(1, -1, 1, 1, 1).float()
        self.scale = scale.to(device).view(1, -1, 1, 1, 1).float()

    @torch.no_grad()
    def encode(self, x01: torch.Tensor) -> torch.Tensor:
        """[B, 1, 64³] occupancy in [0,1] -> normalized latents (fp32).

        Uses the posterior mean (deterministic — resumed slices must see the
        same latents for the same samples).
        """
        mu, _ = self.vae.encode(x01.float().to(self.device))
        return (mu - self.shift) * self.scale

    @torch.no_grad()
    def decode(self, z_norm: torch.Tensor) -> torch.Tensor:
        """Normalized latents -> [0,1] occupancy probabilities (sigmoid inside)."""
        z = z_norm.float().to(self.device) / self.scale + self.shift
        return self.vae.decode(z)

    @staticmethod
    @torch.no_grad()
    def compute_stats(vae: VAE3D, volumes: torch.Tensor, batch_size: int = 32,
                      device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-channel (shift, scale) over a deterministic sample prefix.

        Pure function of (VAE weights, volumes) — every resumed slice
        recomputes identical values, so the unconditional config.json rewrite
        stays idempotent.
        """
        vae = vae.to(device).eval()
        mus = []
        for i in range(0, len(volumes), batch_size):
            mu, _ = vae.encode(volumes[i:i + batch_size].float().to(device))
            mus.append(mu.cpu())
        mu = torch.cat(mus)
        shift = mu.mean(dim=(0, 2, 3, 4))
        std = mu.std(dim=(0, 2, 3, 4)).clamp_min(1e-6)
        return shift, 1.0 / std


def load_vae(checkpoint_path: Path, device: str = "cpu") -> Tuple[VAE3D, dict]:
    """Rebuild a VAE3D from a trainer checkpoint + sibling config.json
    (same recipe as load_diffusion_pipeline for trainer checkpoints)."""
    from deepsculpt.core.latent.loader import find_config

    checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    config = find_config(Path(checkpoint_path))
    vae = VAE3D(
        in_channels=config.get("in_channels", 1),
        latent_channels=config.get("latent_channels", 4),
        base_channels=config.get("base_channels", 32),
    ).to(device)
    state = checkpoint.get("model_state_dict", checkpoint)
    vae.load_state_dict(state)
    vae.eval()
    return vae, config


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
