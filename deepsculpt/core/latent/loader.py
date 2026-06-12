"""
Checkpoint loading for latent-space navigation.

Handles both checkpoint formats DeepSculpt produces:
  - bare state_dicts: generator_final.pt / ema_generator_final.pt written by
    `train-gan` next to config.json (deepsculpt/main.py)
  - trainer checkpoints: checkpoints/checkpoint_epoch_N.pth dicts with
    generator_state_dict / ema_generator_state_dict keys
    (core/training/gan_trainer.py save_checkpoint)

Architecture config is NOT embedded in the weights — it lives in the sibling
config.json, which this loader resolves from the checkpoint's parent (or
grandparent, for the checkpoints/ subdirectory case).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch

logger = logging.getLogger(__name__)


@dataclass
class LoadedGenerator:
    generator: torch.nn.Module
    config: Dict[str, Any]
    noise_dim: int
    void_dim: int
    source: str  # "bare_state_dict" | "trainer_checkpoint" | "trainer_checkpoint_ema"


def find_config(checkpoint_path: Path) -> Dict[str, Any]:
    """Locate and parse the config.json next to (or one level above) a checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    for candidate in (checkpoint_path.parent / "config.json",
                      checkpoint_path.parent.parent / "config.json"):
        if candidate.exists():
            with open(candidate) as f:
                return json.load(f)
    raise FileNotFoundError(
        f"No config.json found next to {checkpoint_path} (looked in parent and "
        "grandparent directories) — cannot rebuild the generator architecture."
    )


def load_generator(
    checkpoint_path: Path,
    device: str = "cpu",
    prefer_ema: bool = True,
) -> LoadedGenerator:
    """Rebuild a generator from a checkpoint + sibling config.json."""
    from deepsculpt.core.models.model_factory import PyTorchModelFactory

    checkpoint_path = Path(checkpoint_path)
    # weights_only=False: trainer checkpoints pickle a TrainingConfig dataclass
    obj = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(obj, dict) and "generator_state_dict" in obj:
        if prefer_ema and obj.get("ema_generator_state_dict") is not None:
            state_dict = obj["ema_generator_state_dict"]
            source = "trainer_checkpoint_ema"
        else:
            state_dict = obj["generator_state_dict"]
            source = "trainer_checkpoint"
    else:
        state_dict = obj
        source = "bare_state_dict"

    config = find_config(checkpoint_path)

    factory = PyTorchModelFactory(device=device)
    generator = factory.create_gan_generator(
        model_type=config["model_type"],
        void_dim=config["void_dim"],
        noise_dim=config["noise_dim"],
        color_mode=config.get("color_mode", 0),
        sparse=config.get("sparse", False),
        **({"gen_channels": config["gen_channels"]} if "gen_channels" in config else {}),
    )
    generator.load_state_dict(state_dict)
    generator.eval()

    logger.info("Loaded %s generator from %s (%s)", config["model_type"], checkpoint_path, source)
    return LoadedGenerator(
        generator=generator,
        config=config,
        noise_dim=config["noise_dim"],
        void_dim=config["void_dim"],
        source=source,
    )


def load_diffusion_pipeline(
    checkpoint_path: Path,
    device: str = "cpu",
    sampler: str = "ddim",
    num_steps: int = 10,
    guidance_scale: float = 1.0,
):
    """Rebuild a FastSamplingPipeline from a diffusion_final.pt checkpoint
    (same recipe as the sample-diffusion CLI). Returns (pipeline, config).

    Only deterministic samplers make sense for noise-space walks — the
    caller should reject 'ddpm'.
    """
    from deepsculpt.core.models.model_factory import PyTorchModelFactory
    from deepsculpt.core.models.diffusion.pipeline import FastSamplingPipeline

    checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    config = checkpoint["config"]

    factory = PyTorchModelFactory(device=device)
    model = factory.create_diffusion_model(
        model_type="unet3d",
        void_dim=config["void_dim"],
        in_channels=config.get("num_channels", 1),
        out_channels=config.get("num_channels", 1),
        timesteps=config.get("timesteps", 1000),
        sparse=config.get("sparse", False),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    noise_scheduler = checkpoint["noise_scheduler"]
    if hasattr(noise_scheduler, "device"):
        noise_scheduler.device = device
    if hasattr(noise_scheduler, "_to_device"):
        noise_scheduler._to_device()

    pipeline = FastSamplingPipeline(
        model=model,
        noise_scheduler=noise_scheduler,
        device=device,
        guidance_scale=guidance_scale,
        num_inference_steps=num_steps,
        scheduler_type=sampler,
    )
    logger.info("Loaded diffusion pipeline from %s (%s, %d steps)",
                checkpoint_path, sampler, num_steps)
    return pipeline, config
