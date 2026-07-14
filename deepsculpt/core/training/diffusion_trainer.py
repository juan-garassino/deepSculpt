"""
Diffusion trainer for DeepSculpt PyTorch implementation.

This module provides specialized training infrastructure for diffusion models
with support for various prediction types, conditioning, and advanced sampling techniques.
"""

import os
import json
import time
import logging
from typing import Dict, Any, Optional, Tuple, List, Union, Callable
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import numpy as np
from datetime import datetime

from .base_trainer import BaseTrainer, TrainingConfig
from .training_metrics import TrainingMetrics
from ..models.diffusion.noise_scheduler import NoiseScheduler
from ..models.diffusion.pipeline import Diffusion3DPipeline


class DiffusionTrainer(BaseTrainer):
    """
    Specialized trainer for diffusion models.
    
    Features:
    - Multiple prediction types (epsilon, sample, v_prediction)
    - Classifier-free guidance training
    - EMA model for better sample quality
    - Advanced loss functions and sampling techniques
    - Conditioning support
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: TrainingConfig,
        noise_scheduler: NoiseScheduler,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda",
        prediction_type: str = "epsilon",  # "epsilon", "sample", "v_prediction"
        conditioning_key: Optional[str] = None,
        conditioning_dropout: float = 0.1,
        use_ema: bool = True,
        ema_decay: float = 0.9999,
        loss_type: str = "mse",  # "mse", "l1", "huber"
        min_snr_gamma: Optional[float] = None,  # None/0 = off; 5.0 = Min-SNR-5 weighting
        codec: Optional[Any] = None,  # LatentCodec => train in VAE latent space
        latent_input_fn: Optional[Any] = None  # batch dict -> [0,1] volume to encode (RGBA colour)
    ):
        """
        Initialize diffusion trainer.
        
        Args:
            model: Diffusion model (e.g., UNet3D)
            optimizer: Model optimizer
            config: Training configuration
            noise_scheduler: Noise scheduler for diffusion process
            scheduler: Learning rate scheduler
            device: Device for training
            prediction_type: Type of model prediction
            conditioning_key: Key for conditioning information in data
            conditioning_dropout: Dropout rate for classifier-free guidance
            use_ema: Whether to use EMA model
            ema_decay: EMA decay rate
            loss_type: Type of loss function
        """
        super().__init__(model, optimizer, config, scheduler, device)
        
        self.noise_scheduler = noise_scheduler
        self.prediction_type = prediction_type
        self.conditioning_key = conditioning_key
        self.conditioning_dropout = conditioning_dropout
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.loss_type = loss_type
        self.min_snr_gamma = min_snr_gamma if min_snr_gamma else None
        # Latent mode: frozen VAE codec held as a trainer attribute (never a
        # UNet submodule) so EMA copying, the optimizer, and grad clipping all
        # keep operating on the UNet alone. latent_input_fn builds the volume to
        # encode (RGBA colour) from the raw batch; None => encode mono structure.
        self.codec = codec
        self.latent_input_fn = latent_input_fn
        
        # Diffusion-specific metrics
        self.metrics.update({
            'diffusion_loss': [],
            'mse_loss': [],
            'l1_loss': [],
            'perceptual_loss': [],
            'timestep_loss': {},  # Loss per timestep
            'conditioning_accuracy': []
        })
        
        # EMA model for better sample quality
        self.ema_model = self._create_ema_model() if use_ema else None
        
        # Create diffusion pipeline for sampling
        self.pipeline = Diffusion3DPipeline(
            model=self.ema_model if self.ema_model else self.model,
            noise_scheduler=noise_scheduler,
            device=device,
            prediction_type=prediction_type
        )
        
        # Initialize metrics tracker
        self.metrics_tracker = TrainingMetrics()
        self.last_epoch_metrics: Dict[str, float] = {}
        
        self.logger.info(f"Diffusion trainer initialized with {prediction_type} prediction")
    
    def _create_ema_model(self) -> nn.Module:
        """Create EMA version of the model."""
        try:
            # Try to create a copy of the model
            ema_model = type(self.model)(**self.model.init_kwargs)
        except:
            # Fallback: create a deep copy
            import copy
            ema_model = copy.deepcopy(self.model)
        
        ema_model.load_state_dict(self.model.state_dict())
        ema_model.eval()
        ema_model = ema_model.to(self.device)
        return ema_model
    
    def _update_ema_model(self):
        """Update EMA model parameters."""
        if self.ema_model is None:
            return
        
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
    
    def compute_loss(
        self,
        model_output: torch.Tensor,
        target: torch.Tensor,
        timesteps: torch.Tensor,
        sample: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute diffusion loss based on prediction type.
        
        Args:
            model_output: Model prediction
            target: Target tensor
            timesteps: Timesteps for each sample
            sample: Original sample
            conditioning: Optional conditioning information
            
        Returns:
            Dictionary of computed losses
        """
        losses = {}

        # Main loss based on prediction type
        if self.min_snr_gamma:
            # Min-SNR weighting (Hang et al. 2023): per-sample loss weighted by
            # min(SNR, gamma)/SNR (epsilon) or min(SNR, gamma)/(SNR+1) (v-pred)
            # so easy high-noise timesteps stop dominating the objective.
            if self.loss_type == "mse":
                per_elem = F.mse_loss(model_output, target, reduction="none")
            elif self.loss_type == "l1":
                per_elem = F.l1_loss(model_output, target, reduction="none")
            elif self.loss_type == "huber":
                per_elem = F.huber_loss(model_output, target, reduction="none")
            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")
            per_sample = per_elem.mean(dim=list(range(1, per_elem.dim())))
            alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(timesteps.device)
            snr = alphas_cumprod[timesteps] / (1 - alphas_cumprod[timesteps])
            if self.prediction_type == "v_prediction":
                weight = torch.clamp(snr, max=self.min_snr_gamma) / (snr + 1)
            else:
                weight = torch.clamp(snr, max=self.min_snr_gamma) / snr
            main_loss = (weight * per_sample).mean()
        elif self.loss_type == "mse":
            main_loss = F.mse_loss(model_output, target)
        elif self.loss_type == "l1":
            main_loss = F.l1_loss(model_output, target)
        elif self.loss_type == "huber":
            main_loss = F.huber_loss(model_output, target)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        losses['diffusion_loss'] = main_loss
        losses['mse_loss'] = F.mse_loss(model_output, target)
        losses['l1_loss'] = F.l1_loss(model_output, target)
        
        # Timestep-specific losses for analysis
        unique_timesteps = torch.unique(timesteps)
        for t in unique_timesteps:
            mask = timesteps == t
            if mask.any():
                t_loss = F.mse_loss(model_output[mask], target[mask])
                losses[f'timestep_{t.item()}_loss'] = t_loss
        
        # Conditioning accuracy if applicable
        if conditioning is not None and hasattr(self.model, 'conditioning_accuracy'):
            cond_acc = self.model.conditioning_accuracy(model_output, conditioning)
            losses['conditioning_accuracy'] = cond_acc
        
        return losses
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Execute a single training step.
        
        Args:
            batch: Batch of training data
            
        Returns:
            Dictionary of step metrics
        """
        # Extract data from batch
        if isinstance(batch, dict):
            structure = batch.get('data', batch.get('structure', batch.get('x', None)))
            conditioning = batch.get(self.conditioning_key) if self.conditioning_key else None
            class_labels = batch.get("class_labels")
            
            if structure is None:
                raise ValueError("Could not find data in batch")
            
            structure = structure.to(self.device)
            
            # Convert to proper PyTorch format: [batch, channels, depth, height, width]
            # Need to determine expected channels from the model
            expected_channels = getattr(self.model, 'in_channels', 1)
            
            if structure.dim() == 4:  # [batch, depth, height, width]
                if expected_channels == 1:  # Monochrome mode
                    # Add single channel dimension at position 1 (PyTorch format)
                    x_0 = structure.unsqueeze(1)
                elif expected_channels == 6:  # Color mode with 6 channels
                    # For color mode, we need to create 6 channels from structure and colors
                    colors = batch.get("colors", structure)  # Use structure as fallback
                    colors = colors.to(self.device)
                    # Create 6 channels - this is a simplified approach
                    x_0 = torch.stack([
                        structure, colors, structure, colors, structure, colors
                    ], dim=1)  # Stack along channel dimension
                elif expected_channels == 3:  # Alternative color mode with 3 channels
                    # Create 3 channels from structure and colors
                    colors = batch.get("colors", structure)
                    colors = colors.to(self.device)
                    combined = (structure + colors) / 2
                    x_0 = torch.stack([structure, colors, combined], dim=1)
                else:
                    # Default: just add single channel
                    x_0 = structure.unsqueeze(1)
            elif structure.dim() == 5:  # Already has channel dimension
                x_0 = structure
            else:
                x_0 = structure
            
            # Convert to float if needed (diffusion models expect float tensors)
            if x_0.dtype != torch.float32:
                x_0 = x_0.float()
                
        else:
            x_0 = batch.to(self.device)
            conditioning = None
            
            # Convert to float if needed
            if x_0.dtype != torch.float32:
                x_0 = x_0.float()
        
        if self.codec is not None:
            # Latent mode: encode [0,1] volumes to normalized latents (fp32,
            # no_grad, before autocast). Everything downstream — _x0_shape,
            # fixed snapshot noise, slerp walk anchors — follows the latent
            # shape automatically. latent_input_fn builds the RGBA colour
            # volume; without it we encode the mono structure.
            vol = self.latent_input_fn(batch) if self.latent_input_fn is not None else x_0
            x_0 = self.codec.encode(vol)
        else:
            # Zero-center binary volumes to [-1, 1]: the DDIM/DDPM samplers
            # clamp pred_original_sample to [-1, 1] (symmetric-data
            # convention), and training on raw {0,1} leaves a +mean bias the
            # model burns dozens of epochs unlearning — samples came out ~3x
            # too dense (occ 0.4 vs 0.12). pipeline.sample() maps back to [0,1].
            x_0 = x_0 * 2.0 - 1.0

        if conditioning is not None:
            conditioning = conditioning.to(self.device)
        if class_labels is not None:
            class_labels = class_labels.to(self.device)
        
        batch_size = x_0.shape[0]
        # Remember the per-sample shape so snapshot noise matches the data
        # layout exactly (sample_and_log's hardcoded shape predates this).
        self._x0_shape = tuple(x_0.shape[1:])

        # Sample random timesteps
        timesteps = torch.randint(
            0, self.noise_scheduler.timesteps, (batch_size,), device=self.device, dtype=torch.long
        )
        
        # Add noise to samples
        noise = torch.randn_like(x_0)
        x_t = self.noise_scheduler.add_noise(x_0, noise, timesteps)
        
        # Apply conditioning dropout for classifier-free guidance
        if conditioning is not None and self.conditioning_dropout > 0:
            dropout_mask = torch.rand(batch_size, device=self.device) < self.conditioning_dropout
            conditioning = conditioning.clone()
            conditioning[dropout_mask] = 0  # Zero out conditioning for dropped samples
            if class_labels is not None:
                class_labels = class_labels.clone()
                class_labels[dropout_mask] = 0
        
        # Forward pass
        self.optimizer.zero_grad()
        
        if self.config.mixed_precision:
            # torch.autocast like the GAN trainer — the legacy
            # torch.cuda.amp.autocast() stopped casting on torch 2.13, so the
            # whole forward ran fp32 and SDPA fell back to the math kernel
            # (materializes B*H*4096^2 attention matrices -> OOM on the L4).
            # bf16 over fp16 for the same overflow reasons as the GAN.
            device_type = "cuda" if "cuda" in str(self.device) else "cpu"
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                # Get model prediction
                if conditioning is not None and "class_labels" in self.model.forward.__code__.co_varnames:
                    model_output = self.model(x_t, timesteps, conditioning, class_labels)
                elif conditioning is not None:
                    model_output = self.model(x_t, timesteps, conditioning)
                elif "class_labels" in self.model.forward.__code__.co_varnames and class_labels is not None:
                    model_output = self.model(x_t, timesteps, None, class_labels)
                else:
                    model_output = self.model(x_t, timesteps)
                
                # Compute target based on prediction type
                if self.prediction_type == "epsilon":
                    target = noise
                elif self.prediction_type == "sample":
                    target = x_0
                elif self.prediction_type == "v_prediction":
                    target = self.noise_scheduler.get_velocity(x_0, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type: {self.prediction_type}")
                
                # Compute losses
                losses = self.compute_loss(model_output, target, timesteps, x_0, conditioning)
                loss = losses['diffusion_loss']
            
            # Backward pass with mixed precision
            self.scaler.scale(loss).backward()
            
            if self.config.gradient_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Get model prediction
            if conditioning is not None and "class_labels" in self.model.forward.__code__.co_varnames:
                model_output = self.model(x_t, timesteps, conditioning, class_labels)
            elif conditioning is not None:
                model_output = self.model(x_t, timesteps, conditioning)
            elif "class_labels" in self.model.forward.__code__.co_varnames and class_labels is not None:
                model_output = self.model(x_t, timesteps, None, class_labels)
            else:
                model_output = self.model(x_t, timesteps)
            
            # Compute target based on prediction type
            if self.prediction_type == "epsilon":
                target = noise
            elif self.prediction_type == "sample":
                target = x_0
            elif self.prediction_type == "v_prediction":
                target = self.noise_scheduler.get_velocity(x_0, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type: {self.prediction_type}")
            
            # Compute losses
            losses = self.compute_loss(model_output, target, timesteps, x_0, conditioning)
            loss = losses['diffusion_loss']
            
            # Backward pass
            loss.backward()
            
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            
            self.optimizer.step()
        
        # Update EMA model
        if self.ema_model is not None:
            self._update_ema_model()
        
        # Convert losses to float for logging
        step_metrics = {key: value.item() if torch.is_tensor(value) else value 
                       for key, value in losses.items()}
        
        # Update metrics tracker
        self.metrics_tracker.update_step_metrics(step_metrics)
        
        return step_metrics
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data loader
            
        Returns:
            Dictionary of epoch metrics
        """
        self.model.train()
        
        epoch_metrics = {
            'diffusion_loss': [],
            'mse_loss': [],
            'l1_loss': []
        }
        
        for batch_idx, batch in enumerate(dataloader):
            # Training step
            step_metrics = self.train_step(batch)
            
            # Accumulate metrics
            for key, value in step_metrics.items():
                if key in epoch_metrics:
                    epoch_metrics[key].append(value)
                elif key.startswith('timestep_'):
                    # Handle timestep-specific losses
                    if key not in epoch_metrics:
                        epoch_metrics[key] = []
                    epoch_metrics[key].append(value)
            
            # Log step metrics
            if batch_idx % self.config.log_freq == 0:
                self.log_metrics(step_metrics, self.global_step, "train_step")
                self.logger.info(
                    f"Epoch {self.current_epoch}, Batch {batch_idx}: "
                    f"Diffusion Loss: {step_metrics.get('diffusion_loss', 0):.4f}, "
                    f"MSE Loss: {step_metrics.get('mse_loss', 0):.4f}"
                )
            
            self.global_step += 1
        
        # Calculate epoch averages
        avg_metrics = {}
        for key, values in epoch_metrics.items():
            if values:  # Only include metrics that have values
                avg_metrics[key] = np.mean(values)
        
        # Update learning rate scheduler
        if self.scheduler:
            self.scheduler.step()
        
        # Update metrics tracker
        self.metrics_tracker.update_epoch_metrics(avg_metrics)
        self.last_epoch_metrics = avg_metrics
        
        return avg_metrics
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        val_metrics = {
            'diffusion_loss': [],
            'mse_loss': [],
            'l1_loss': []
        }
        
        with torch.no_grad():
            for batch in dataloader:
                # Extract data from batch
                if isinstance(batch, dict):
                    x_0 = batch.get('data', batch.get('structure', batch.get('x', None)))
                    conditioning = batch.get(self.conditioning_key) if self.conditioning_key else None
                else:
                    x_0 = batch
                    conditioning = None
                
                if x_0 is None:
                    continue
                
                x_0 = x_0.to(self.device)
                if conditioning is not None:
                    conditioning = conditioning.to(self.device)
                
                batch_size = x_0.shape[0]
                
                # Sample random timesteps
                timesteps = torch.randint(
                    0, self.noise_scheduler.timesteps, (batch_size,), device=self.device, dtype=torch.long
                )
                
                # Add noise to samples
                noise = torch.randn_like(x_0)
                x_t = self.noise_scheduler.add_noise(x_0, noise, timesteps)
                
                # Get model prediction
                if conditioning is not None:
                    model_output = self.model(x_t, timesteps, conditioning)
                else:
                    model_output = self.model(x_t, timesteps)
                
                # Compute target based on prediction type
                if self.prediction_type == "epsilon":
                    target = noise
                elif self.prediction_type == "sample":
                    target = x_0
                elif self.prediction_type == "v_prediction":
                    target = self.noise_scheduler.get_velocity(x_0, noise, timesteps)
                
                # Compute losses
                losses = self.compute_loss(model_output, target, timesteps, x_0, conditioning)
                
                # Accumulate validation metrics
                for key, value in losses.items():
                    if key in val_metrics:
                        val_metrics[key].append(value.item() if torch.is_tensor(value) else value)
        
        return {key: np.mean(values) for key, values in val_metrics.items() if values}
    
    def _snapshot_sample_stats(self, samples: torch.Tensor) -> Dict[str, float]:
        # Diffusion outputs are continuous; occupancy judged at the 0.5
        # threshold used by the volume exporters. RGBA samples carry colour in
        # channels 1-3 (mostly >0.5) — occupancy is channel 0 (alpha) only.
        s = samples.detach()
        if s.dim() == 5 and s.shape[1] > 1:
            s = s[:, 0:1]
        occupancy = (s > 0.5).float().reshape(s.shape[0], -1).mean(dim=1)
        return {
            "mean_occupancy": float(occupancy.mean().item()),
            "min_occupancy": float(occupancy.min().item()),
            "max_occupancy": float(occupancy.max().item()),
        }

    def _fixed_snapshot_noise(self) -> torch.Tensor:
        """Six CPU-seeded noise tensors: [0:4] are the fixed sample vectors,
        [4:6] the walk anchors. Deterministic across restarts and slices."""
        if getattr(self, "_snapshot_noise", None) is None:
            g = torch.Generator().manual_seed(1234)
            self._snapshot_noise = torch.randn(6, *self._x0_shape, generator=g)
        return self._snapshot_noise

    def _after_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]],
        is_best: bool,
    ) -> None:
        if (epoch + 1) % max(1, self.config.snapshot_freq) != 0:
            return
        if getattr(self, "_x0_shape", None) is None:
            return
        try:
            self._save_epoch_snapshot(epoch, train_metrics)
        except Exception:
            self.logger.exception("Epoch snapshot failed (training continues)")
        finally:
            # The snapshot sampling (fixed-vector samples + the periodic
            # noise-space walk) is the per-epoch memory high-water mark; on the
            # 32Gi L4 the latent runs OOM-killed (exit 137) right at the walk
            # epoch. Drop the sampling pipeline + reclaim CPU/GPU memory so the
            # baseline doesn't creep across epochs.
            import gc
            self._snap_pipeline = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _snapshot_pipeline(self):
        """Deterministic DDIM pipeline for epoch snapshots. The base
        Diffusion3DPipeline/NoiseScheduler pair is the training-side forward
        process; DDIM (eta=0) is the proven sampling path (sample-diffusion
        CLI) and makes fixed-noise snapshots bit-comparable across epochs."""
        from deepsculpt.core.models.diffusion.pipeline import (
            FastSamplingPipeline,
            LatentFastSamplingPipeline,
        )

        if getattr(self, "_snap_pipeline", None) is None:
            if self.codec is not None:
                self._snap_pipeline = LatentFastSamplingPipeline(
                    codec=self.codec,
                    model=self.ema_model if self.ema_model else self.model,
                    noise_scheduler=self.noise_scheduler,
                    device=self.device,
                    prediction_type=self.prediction_type,
                    num_inference_steps=25,
                    scheduler_type="ddim",
                )
            else:
                self._snap_pipeline = FastSamplingPipeline(
                    model=self.ema_model if self.ema_model else self.model,
                    noise_scheduler=self.noise_scheduler,
                    device=self.device,
                    prediction_type=self.prediction_type,
                    num_inference_steps=25,
                    scheduler_type="ddim",
                )
        self._snap_pipeline.model = self.ema_model if self.ema_model else self.model
        return self._snap_pipeline

    def _save_epoch_snapshot(self, epoch: int, train_metrics: Dict[str, float]) -> None:
        from deepsculpt.core.latent.ops import slerp

        os.makedirs(self.config.snapshot_dir, exist_ok=True)
        noise = self._fixed_snapshot_noise()
        pipeline = self._snapshot_pipeline()

        snapshot_stem = Path(self.config.snapshot_dir) / f"epoch_{epoch + 1:03d}"
        with torch.no_grad():
            samples = pipeline.sample(
                shape=(4, *self._x0_shape),
                num_inference_steps=25,
                init_noise=noise[:4].to(self.device),
            ).detach().cpu()
        torch.save(samples.half(), snapshot_stem.with_suffix(".pt"))
        snapshot_stats = self._snapshot_sample_stats(samples)

        # Noise-space walk is ~40 extra UNet batches — only every 5th snapshot.
        # Latent runs (self.codec set) SKIP it: on the 32Gi L4 this walk is the
        # per-epoch memory high-water mark that OOM-killed (exit 137) every
        # latent slice at epoch 5. Latent walks are produced on demand via
        # MODE=render latent-walk instead, so nothing is lost.
        walk_stats = None
        if self.codec is None and (epoch + 1) % (5 * max(1, self.config.snapshot_freq)) == 0:
            path = slerp(noise[4], noise[5], 8).to(self.device)
            with torch.no_grad():
                walk = torch.cat([
                    pipeline.sample(
                        shape=(1, *self._x0_shape),
                        num_inference_steps=25,
                        init_noise=path[i:i + 1],
                    ).detach().cpu()
                    for i in range(path.shape[0])
                ])
            torch.save(walk.half(), snapshot_stem.with_name(f"walk_epoch_{epoch + 1:03d}.pt"))
            walk_stats = self._snapshot_sample_stats(walk)

        with open(snapshot_stem.with_suffix(".json"), "w") as f:
            json.dump(
                {
                    "epoch": epoch + 1,
                    "train_metrics": {k: float(v) for k, v in train_metrics.items()
                                      if isinstance(v, (int, float))},
                    "sample_stats": snapshot_stats,
                    "walk_stats": walk_stats,
                },
                f,
                indent=2,
            )

    def sample_and_log(self, num_samples: int = 8, conditioning: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate samples and log them.
        
        Args:
            num_samples: Number of samples to generate
            conditioning: Optional conditioning information
            
        Returns:
            Generated samples
        """
        # Use EMA model if available
        model_to_use = self.ema_model if self.ema_model else self.model
        
        # Update pipeline model
        self.pipeline.model = model_to_use
        
        # Generate samples
        shape = (num_samples, 64, 64, 64, 6)  # Default shape, should be configurable
        samples = self.pipeline.sample(
            shape=shape,
            conditioning=conditioning,
            num_inference_steps=50
        )
        
        return samples
    
    def save_checkpoint(self, path: str, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """
        Save diffusion training checkpoint.
        
        Args:
            path: Path to save checkpoint
            epoch: Current epoch
            metrics: Current metrics
            is_best: Whether this is the best checkpoint
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'best_loss': self.best_loss,
            'prediction_type': self.prediction_type,
            'conditioning_key': self.conditioning_key,
            'conditioning_dropout': self.conditioning_dropout,
            'loss_type': self.loss_type,
            'noise_scheduler_state': {
                'schedule_type': self.noise_scheduler.schedule_type,
                'timesteps': self.noise_scheduler.timesteps,
                'beta_start': self.noise_scheduler.beta_start,
                'beta_end': self.noise_scheduler.beta_end,
            }
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        if self.ema_model:
            checkpoint['ema_model_state_dict'] = self.ema_model.state_dict()
            checkpoint['ema_decay'] = self.ema_decay

        if self.codec is not None:
            checkpoint['latent_shift'] = self.codec.shift.flatten().cpu()
            checkpoint['latent_scale'] = self.codec.scale.flatten().cpu()

        torch.save(checkpoint, path)
        
        if is_best:
            best_path = path.replace('.pth', '_best.pth')
            torch.save(checkpoint, best_path)
        
        self.logger.info(f"Diffusion checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """
        Load diffusion training checkpoint.
        
        Args:
            path: Path to checkpoint file
            
        Returns:
            Loaded checkpoint data
        """
        # weights_only=False: our own checkpoints pickle TrainingConfig
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        if self.ema_model and 'ema_model_state_dict' in checkpoint:
            self.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
            self.ema_decay = checkpoint.get('ema_decay', self.ema_decay)
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        # Update training parameters
        self.prediction_type = checkpoint.get('prediction_type', self.prediction_type)
        self.conditioning_key = checkpoint.get('conditioning_key', self.conditioning_key)
        self.conditioning_dropout = checkpoint.get('conditioning_dropout', self.conditioning_dropout)
        self.loss_type = checkpoint.get('loss_type', self.loss_type)

        # Latent stats are a pure function of (VAE weights, dataset prefix) —
        # a resumed slice recomputing different values means the data pipeline
        # changed under the run. Fail loudly, never train on shifted latents.
        if self.codec is not None and 'latent_shift' in checkpoint:
            for name, ours in (('latent_shift', self.codec.shift.flatten().cpu()),
                               ('latent_scale', self.codec.scale.flatten().cpu())):
                saved = checkpoint[name].float()
                if not torch.allclose(saved, ours.float(), atol=1e-4):
                    raise RuntimeError(
                        f"{name} mismatch on resume: checkpoint {saved.tolist()} vs "
                        f"recomputed {ours.tolist()} — VAE or dataset prefix changed")
        
        self.logger.info(f"Diffusion checkpoint loaded: {path}")
        return checkpoint
    
    def get_training_info(self) -> Dict[str, Any]:
        """Get comprehensive training information."""
        info = super().get_training_info()
        info.update({
            "prediction_type": self.prediction_type,
            "conditioning_key": self.conditioning_key,
            "conditioning_dropout": self.conditioning_dropout,
            "use_ema": self.use_ema,
            "ema_decay": self.ema_decay,
            "loss_type": self.loss_type,
            "noise_scheduler_type": self.noise_scheduler.__class__.__name__,
            "timesteps": self.noise_scheduler.timesteps,
            "last_epoch_metrics": self.last_epoch_metrics,
        })
        return info
