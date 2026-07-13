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

    # Two on-disk shapes: diffusion_final.pt (post-training export, embeds a
    # plain-dict config + pickled scheduler) and trainer checkpoint_epoch_N.pth
    # (cloud slices die at the task timeout, so the final export may never
    # exist — rebuild from the sibling config.json + noise_scheduler_state).
    is_trainer_ckpt = "noise_scheduler" not in checkpoint
    if is_trainer_ckpt:
        config = find_config(Path(checkpoint_path))
        state_dict = checkpoint.get("ema_model_state_dict") or checkpoint["model_state_dict"]
    else:
        config = checkpoint["config"]
        state_dict = checkpoint["model_state_dict"]

    factory = PyTorchModelFactory(device=device)
    model = factory.create_diffusion_model(
        model_type="unet3d",
        void_dim=config["void_dim"],
        in_channels=config.get("num_channels", 1),
        out_channels=config.get("num_channels", 1),
        timesteps=config.get("timesteps", 1000),
        sparse=config.get("sparse", False),
        model_channels=config.get("model_channels", 128),
        # Older config.json lacks these — factory defaults reproduce the
        # historical architecture exactly.
        num_res_blocks=config.get("num_res_blocks", 2),
        channel_mult=config.get("channel_mult", [1, 2, 4, 8]),
        attention_resolutions=config.get("attention_resolutions", [16, 8]),
        num_heads=config.get("num_heads", 8),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    if is_trainer_ckpt:
        from deepsculpt.core.models.diffusion.noise_scheduler import NoiseScheduler

        ns = checkpoint.get("noise_scheduler_state", {})
        noise_scheduler = NoiseScheduler(
            schedule_type=ns.get("schedule_type", config.get("noise_schedule", "linear")),
            timesteps=ns.get("timesteps", config.get("timesteps", 1000)),
            beta_start=ns.get("beta_start", config.get("beta_start", 0.0001)),
            beta_end=ns.get("beta_end", config.get("beta_end", 0.02)),
            device=device,
        )
    else:
        noise_scheduler = checkpoint["noise_scheduler"]
    if hasattr(noise_scheduler, "device"):
        noise_scheduler.device = device
    if hasattr(noise_scheduler, "_to_device"):
        noise_scheduler._to_device()

    latent_cfg = config.get("latent") or {}
    if latent_cfg.get("enabled"):
        # Latent run: rebuild the codec from the self-contained run dir
        # (autoencoder.pt travels with the run) and return the decoding
        # pipeline — callers keep receiving [0,1] occupancy volumes.
        from deepsculpt.core.models.autoencoder import VAE3D, LatentCodec
        from deepsculpt.core.models.diffusion.pipeline import LatentFastSamplingPipeline

        ae_path = None
        for parent in (Path(checkpoint_path).parent, Path(checkpoint_path).parent.parent):
            if (parent / "autoencoder.pt").exists():
                ae_path = parent / "autoencoder.pt"
                break
        if ae_path is None:
            raise FileNotFoundError(
                f"latent run but no autoencoder.pt beside {checkpoint_path}")
        vae = VAE3D(
            in_channels=latent_cfg.get("vae_in_channels", 1),
            latent_channels=latent_cfg.get("latent_channels", 4),
            base_channels=latent_cfg.get("vae_base_channels", 32),
        ).to(device)
        vae.load_state_dict(torch.load(ae_path, map_location=device, weights_only=False))
        codec = LatentCodec(
            vae,
            torch.tensor(latent_cfg["shift"]),
            torch.tensor(latent_cfg["scale"]),
            device=device,
        )
        pipeline = LatentFastSamplingPipeline(
            codec=codec,
            model=model,
            noise_scheduler=noise_scheduler,
            device=device,
            prediction_type=config.get("prediction_type", "epsilon"),
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            scheduler_type=sampler,
        )
    else:
        pipeline = FastSamplingPipeline(
            model=model,
            noise_scheduler=noise_scheduler,
            device=device,
            prediction_type=config.get("prediction_type", "epsilon"),
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            scheduler_type=sampler,
        )
    logger.info("Loaded diffusion pipeline from %s (%s, %d steps)",
                checkpoint_path, sampler, num_steps)
    return pipeline, config
