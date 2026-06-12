"""
Semantic direction discovery in GAN latent space.

Two methods:
  - "ganspace": PCA on the first linear layer's activations W·z + b over many
    sampled z, mapped back to z-space via W^T (the GANSpace trick). Cheap —
    no generator forward passes through the conv stack.
  - "output-pca": PCA on flattened generator outputs, regressed back to z.
    Slower but captures end-to-end variation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import torch

from .ops import seeded_z

logger = logging.getLogger(__name__)


@dataclass
class LatentDirections:
    directions: torch.Tensor  # (k, noise_dim), unit norm
    explained_variance: List[float]
    method: str  # "ganspace" | "output-pca"
    noise_dim: int


def _pca(data: torch.Tensor, num_components: int) -> tuple[torch.Tensor, List[float]]:
    """PCA via SVD on centered data (n_samples, n_features).
    Returns (components (k, n_features), explained variance ratios)."""
    centered = data - data.mean(dim=0, keepdim=True)
    _, s, vt = torch.linalg.svd(centered, full_matrices=False)
    var = (s ** 2) / max(data.shape[0] - 1, 1)
    ratios = (var / var.sum()).tolist()
    k = min(num_components, vt.shape[0])
    return vt[:k], ratios[:k]


def compute_directions(
    generator: torch.nn.Module,
    noise_dim: int,
    num_samples: int = 2048,
    num_components: int = 10,
    method: str = "ganspace",
    batch_size: int = 32,
    device: str = "cpu",
    seed: int = 0,
) -> LatentDirections:
    zs = seeded_z(seed, noise_dim, batch=num_samples)

    if method == "ganspace":
        linear = next(
            (m for m in generator.modules() if isinstance(m, torch.nn.Linear)), None
        )
        if linear is None:
            raise ValueError("generator has no nn.Linear layer — use method='output-pca'")
        with torch.no_grad():
            activations = torch.nn.functional.linear(
                zs.to(device), linear.weight, linear.bias
            ).cpu()
        components, ratios = _pca(activations, num_components)
        # Map activation-space components back to z-space: d_z = W^T d_act
        dirs = components @ linear.weight.detach().cpu()
    elif method == "output-pca":
        from .ops import batched_generate

        volumes = batched_generate(generator, zs, batch_size=batch_size, device=device)
        flat = volumes.flatten(1)
        components, ratios = _pca(flat, num_components)
        # Regress output-space components back to z-space: solve z ≈ flat @ comp^T
        scores = flat @ components.T  # (n, k)
        dirs, *_ = torch.linalg.lstsq(scores, zs - zs.mean(dim=0, keepdim=True))
        dirs = dirs.T  # (k, noise_dim)
    else:
        raise ValueError(f"unknown method {method!r} (use 'ganspace' or 'output-pca')")

    dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp_min(1e-12)
    logger.info(
        "Computed %d %s directions from %d samples (top variance %.3f)",
        dirs.shape[0], method, num_samples, ratios[0] if ratios else float("nan"),
    )
    return LatentDirections(
        directions=dirs.float(),
        explained_variance=ratios,
        method=method,
        noise_dim=noise_dim,
    )


def save_directions(d: LatentDirections, path: Path) -> None:
    torch.save(
        {
            "directions": d.directions,
            "explained_variance": d.explained_variance,
            "method": d.method,
            "noise_dim": d.noise_dim,
        },
        path,
    )


def load_directions(path: Path) -> LatentDirections:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return LatentDirections(
        directions=obj["directions"],
        explained_variance=obj["explained_variance"],
        method=obj["method"],
        noise_dim=obj["noise_dim"],
    )


def apply_direction(
    z: torch.Tensor, direction: torch.Tensor, alphas: Sequence[float]
) -> torch.Tensor:
    """Move z along a unit direction: returns (len(alphas), *z.shape)."""
    return torch.stack([z + float(a) * direction.reshape(z.shape) for a in alphas])
