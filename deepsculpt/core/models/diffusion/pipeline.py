"""
Diffusion pipeline for 3D sculpture generation in DeepSculpt.

This module provides complete pipelines for training and inference with
diffusion models, including noise scheduling, sampling algorithms, and
conditioning mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List, Union, Callable
import numpy as np
from tqdm import tqdm

from .noise_scheduler import NoiseScheduler, DDIMScheduler, DPMSolverScheduler
from .unet import UNet3D, ConditionalUNet3D


class Diffusion3DPipeline:
    """
    Complete pipeline for 3D diffusion model training and inference.
    
    Handles the full diffusion process including forward noising, reverse denoising,
    and various sampling algorithms for high-quality 3D sculpture generation.
    """
    
    def __init__(
        self,
        model: nn.Module,
        noise_scheduler: NoiseScheduler,
        device: str = "cuda",
        prediction_type: str = "epsilon",
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50
    ):
        """
        Initialize the diffusion pipeline.
        
        Args:
            model: The diffusion model (e.g., UNet3D)
            noise_scheduler: Noise scheduler for the diffusion process
            device: Device to run computations on
            prediction_type: Type of model prediction ("epsilon", "sample", "v_prediction")
            guidance_scale: Scale for classifier-free guidance
            num_inference_steps: Number of steps for inference
        """
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.device = device
        self.prediction_type = prediction_type
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        
        # Move model to device
        self.model = self.model.to(device)
        
        # Statistics tracking
        self.generation_count = 0
        self.total_generation_time = 0.0
    
    def forward_process(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion process - add noise to clean samples.
        
        Args:
            x: Clean samples
            timesteps: Timesteps for each sample
            noise: Optional noise tensor (generated if not provided)
            
        Returns:
            Tuple of (noisy_samples, noise)
        """
        if noise is None:
            noise = torch.randn_like(x)
        
        noisy_samples = self.noise_scheduler.add_noise(x, noise, timesteps)
        return noisy_samples, noise
    
    def reverse_process(
        self,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Single step of reverse diffusion process.
        
        Args:
            x_t: Noisy sample at timestep t
            timestep: Current timestep
            conditioning: Optional conditioning information
            
        Returns:
            Predicted noise or clean sample
        """
        # Get model prediction
        if hasattr(self.model, 'forward') and 'conditioning' in self.model.forward.__code__.co_varnames:
            model_output = self.model(x_t, timestep, conditioning)
        else:
            model_output = self.model(x_t, timestep)
        
        return model_output
    
    def sample(
        self,
        shape: Tuple[int, ...],
        conditioning: Optional[torch.Tensor] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        generator: Optional[torch.Generator] = None,
        return_intermediate: bool = False,
        init_noise: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Generate samples using the reverse diffusion process.

        Args:
            shape: Shape of samples to generate
            conditioning: Optional conditioning information
            num_inference_steps: Number of denoising steps
            guidance_scale: Scale for classifier-free guidance
            generator: Random number generator for reproducibility
            return_intermediate: Whether to return intermediate samples
            init_noise: Custom initial noise tensor (latent-space navigation:
                interpolate between noise tensors, sample each deterministically)

        Returns:
            Generated samples, optionally with intermediate steps
        """
        if num_inference_steps is None:
            num_inference_steps = self.num_inference_steps
        if guidance_scale is None:
            guidance_scale = self.guidance_scale

        # Initialize with provided or random noise
        if init_noise is not None:
            if tuple(init_noise.shape) != tuple(shape):
                raise ValueError(
                    f"init_noise shape {tuple(init_noise.shape)} != requested {tuple(shape)}"
                )
            sample = init_noise.to(self.device)
        else:
            sample = torch.randn(shape, device=self.device, generator=generator)
        
        # Create timestep schedule
        timesteps = torch.linspace(
            self.noise_scheduler.timesteps - 1, 0, num_inference_steps, dtype=torch.long, device=self.device
        )
        
        intermediate_samples = [] if return_intermediate else None
        
        self.model.eval()
        with torch.no_grad():
            for i, t in enumerate(tqdm(timesteps, desc="Sampling")):
                # Expand timestep to batch dimension
                timestep_batch = t.expand(sample.shape[0])
                
                # Get model prediction
                if conditioning is not None and guidance_scale > 1.0:
                    # Classifier-free guidance
                    model_output_cond = self.reverse_process(sample, timestep_batch, conditioning)
                    model_output_uncond = self.reverse_process(sample, timestep_batch, None)
                    
                    model_output = model_output_uncond + guidance_scale * (model_output_cond - model_output_uncond)
                else:
                    model_output = self.reverse_process(sample, timestep_batch, conditioning)
                
                # Compute previous sample. Strided schedules (num_inference_steps <
                # trained timesteps) must tell the scheduler which timestep comes
                # next; DDIM/DPM-Solver default to t-1 otherwise and produce garbage.
                prev_t = int(timesteps[i + 1].item()) if i + 1 < len(timesteps) else -1
                if isinstance(self.noise_scheduler, DDIMScheduler):
                    sample = self.noise_scheduler.step(
                        model_output, t.item(), sample, self.prediction_type, prev_timestep=prev_t
                    )
                elif isinstance(self.noise_scheduler, DPMSolverScheduler):
                    sample = self.noise_scheduler.step(model_output, t.item(), sample, prev_timestep=prev_t)
                elif hasattr(self.noise_scheduler, 'step'):
                    sample = self.noise_scheduler.step(model_output, t.item(), sample, self.prediction_type)
                else:
                    # Fallback to basic DDPM step
                    sample = self._ddpm_step(model_output, t.item(), sample)
                
                if return_intermediate:
                    intermediate_samples.append(sample.clone())
        
        self.generation_count += sample.shape[0]

        sample = self._finalize(sample)
        if return_intermediate:
            intermediate_samples = [self._finalize(s) for s in intermediate_samples]
            return sample, intermediate_samples
        return sample

    def _finalize(self, sample: torch.Tensor) -> torch.Tensor:
        """Map raw sampler output to consumer space — the single hook every
        consumer (trainer snapshots, walks, trajectories, CLI sampling)
        funnels through. Training scales binary volumes to [-1, 1] (see
        DiffusionTrainer.train_step), so the pixel-space base maps back to
        the [0, 1] / threshold-0.5 convention. Latent pipelines override
        this with codec decode. Final clamp covers samplers without a
        per-step clip (DPM-Solver)."""
        return ((sample + 1.0) / 2.0).clamp_(0.0, 1.0)
    
    def _ddpm_step(self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor) -> torch.Tensor:
        """
        Basic DDPM sampling step.
        
        Args:
            model_output: Model prediction
            timestep: Current timestep
            sample: Current sample
            
        Returns:
            Previous sample
        """
        t = timestep
        
        if self.prediction_type == "epsilon":
            # Predict x_0 from epsilon
            pred_original_sample = (
                sample - self.noise_scheduler.sqrt_one_minus_alphas_cumprod[t] * model_output
            ) / self.noise_scheduler.sqrt_alphas_cumprod[t]
        elif self.prediction_type == "sample":
            pred_original_sample = model_output
        else:
            raise ValueError(f"Unsupported prediction type: {self.prediction_type}")
        
        # Clip predicted x_0
        pred_original_sample = torch.clamp(pred_original_sample, -1, 1)
        
        # Compute coefficients
        pred_original_sample_coeff = (
            self.noise_scheduler.sqrt_alphas_cumprod_prev[t] * self.noise_scheduler.betas[t] / 
            (1 - self.noise_scheduler.alphas_cumprod[t])
        )
        current_sample_coeff = (
            self.noise_scheduler.alphas[t] ** 0.5 * (1 - self.noise_scheduler.alphas_cumprod_prev[t]) / 
            (1 - self.noise_scheduler.alphas_cumprod[t])
        )
        
        # Compute predicted previous sample
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample
        
        # Add noise if not the last step
        if t > 0:
            noise = torch.randn_like(sample)
            variance = self.noise_scheduler.posterior_variance[t] ** 0.5
            pred_prev_sample = pred_prev_sample + variance * noise
        
        return pred_prev_sample
    
    def compute_loss(
        self,
        x_0: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None,
        loss_type: str = "mse"
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss for the diffusion model.
        
        Args:
            x_0: Clean samples
            conditioning: Optional conditioning information
            loss_type: Type of loss to compute ("mse", "l1", "huber")
            
        Returns:
            Dictionary of computed losses
        """
        batch_size = x_0.shape[0]
        
        # Sample random timesteps
        timesteps = torch.randint(
            0, self.noise_scheduler.timesteps, (batch_size,), device=self.device, dtype=torch.long
        )
        
        # Add noise to samples
        noise = torch.randn_like(x_0)
        x_t, _ = self.forward_process(x_0, timesteps, noise)
        
        # Get model prediction
        model_output = self.reverse_process(x_t, timesteps, conditioning)
        
        # Compute target based on prediction type
        if self.prediction_type == "epsilon":
            target = noise
        elif self.prediction_type == "sample":
            target = x_0
        elif self.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(x_0, noise, timesteps)
        else:
            raise ValueError(f"Unsupported prediction type: {self.prediction_type}")
        
        # Compute loss
        if loss_type == "mse":
            loss = F.mse_loss(model_output, target)
        elif loss_type == "l1":
            loss = F.l1_loss(model_output, target)
        elif loss_type == "huber":
            loss = F.huber_loss(model_output, target)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
        
        return {
            "loss": loss,
            "mse_loss": F.mse_loss(model_output, target),
            "l1_loss": F.l1_loss(model_output, target),
        }
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the pipeline."""
        return {
            "model_type": self.model.__class__.__name__,
            "scheduler_type": self.noise_scheduler.__class__.__name__,
            "prediction_type": self.prediction_type,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "generation_count": self.generation_count,
            "device": str(self.device),
        }


class ConditionalDiffusion3DPipeline(Diffusion3DPipeline):
    """
    Conditional diffusion pipeline with enhanced conditioning mechanisms.
    """
    
    def __init__(
        self,
        model: nn.Module,
        noise_scheduler: NoiseScheduler,
        device: str = "cuda",
        prediction_type: str = "epsilon",
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        conditioning_dropout: float = 0.1
    ):
        """
        Initialize conditional diffusion pipeline.
        
        Args:
            conditioning_dropout: Probability of dropping conditioning for classifier-free guidance
        """
        super().__init__(model, noise_scheduler, device, prediction_type, guidance_scale, num_inference_steps)
        self.conditioning_dropout = conditioning_dropout
    
    def compute_loss(
        self,
        x_0: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None,
        loss_type: str = "mse"
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss with conditioning dropout for classifier-free guidance.
        
        Args:
            x_0: Clean samples
            conditioning: Conditioning information
            loss_type: Type of loss to compute
            
        Returns:
            Dictionary of computed losses
        """
        # Apply conditioning dropout for classifier-free guidance training
        if conditioning is not None and self.conditioning_dropout > 0:
            batch_size = conditioning.shape[0]
            dropout_mask = torch.rand(batch_size, device=self.device) < self.conditioning_dropout
            conditioning = conditioning.clone()
            conditioning[dropout_mask] = 0  # Zero out conditioning for dropped samples
        
        return super().compute_loss(x_0, conditioning, loss_type)


class FastSamplingPipeline(Diffusion3DPipeline):
    """
    Pipeline optimized for fast sampling with advanced schedulers.
    """
    
    def __init__(
        self,
        model: nn.Module,
        noise_scheduler: NoiseScheduler,
        device: str = "cuda",
        prediction_type: str = "epsilon",
        guidance_scale: float = 7.5,
        num_inference_steps: int = 20,
        scheduler_type: str = "ddim",
        clip_sample: bool = True
    ):
        """
        Initialize fast sampling pipeline.

        Args:
            scheduler_type: Type of fast scheduler ("ddim", "dpm_solver")
            clip_sample: Clamp pred_x0 to [-1, 1] each step. Correct for pixel
                volumes (trained in [-1, 1]); latent pipelines MUST pass False
                or unit-variance latents get crushed every step.
        """
        super().__init__(model, noise_scheduler, device, prediction_type, guidance_scale, num_inference_steps)

        # Create fast scheduler
        if scheduler_type == "ddim":
            self.fast_scheduler = DDIMScheduler(
                schedule_type=noise_scheduler.schedule_type,
                timesteps=noise_scheduler.timesteps,
                beta_start=noise_scheduler.beta_start,
                beta_end=noise_scheduler.beta_end,
                device=device,
                eta=0.0,  # Deterministic sampling
                clip_sample=clip_sample
            )
        elif scheduler_type == "dpm_solver":
            self.fast_scheduler = DPMSolverScheduler(
                schedule_type=noise_scheduler.schedule_type,
                timesteps=noise_scheduler.timesteps,
                beta_start=noise_scheduler.beta_start,
                beta_end=noise_scheduler.beta_end,
                device=device,
                solver_order=2,
                prediction_type=prediction_type
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    
    def sample(
        self,
        shape: Tuple[int, ...],
        conditioning: Optional[torch.Tensor] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        generator: Optional[torch.Generator] = None,
        return_intermediate: bool = False,
        use_fast_scheduler: bool = True,
        init_noise: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Generate samples with optional fast scheduling.

        Args:
            use_fast_scheduler: Whether to use the fast scheduler
            init_noise: Custom initial noise (passed through to the base sampler)

        Returns:
            Generated samples
        """
        if use_fast_scheduler:
            # Temporarily replace scheduler
            original_scheduler = self.noise_scheduler
            self.noise_scheduler = self.fast_scheduler

            try:
                result = super().sample(
                    shape, conditioning, num_inference_steps, guidance_scale, generator,
                    return_intermediate, init_noise=init_noise
                )
            finally:
                # Restore original scheduler
                self.noise_scheduler = original_scheduler

            return result
        else:
            return super().sample(
                shape, conditioning, num_inference_steps, guidance_scale, generator,
                return_intermediate, init_noise=init_noise
            )


class LatentFastSamplingPipeline(FastSamplingPipeline):
    """FastSamplingPipeline that diffuses VAE latents and returns decoded
    [0,1] occupancy volumes. clip_sample is forced off — normalized latents
    are ~unit variance and the [-1,1] per-step clamp would silently crush
    ~a third of their values."""

    def __init__(self, codec, **kwargs):
        kwargs["clip_sample"] = False
        super().__init__(**kwargs)
        self.codec = codec

    def _finalize(self, sample: torch.Tensor) -> torch.Tensor:
        return self.codec.decode(sample)


class ProgressiveDiffusion3DPipeline(Diffusion3DPipeline):
    """
    Progressive diffusion pipeline for multi-resolution generation.
    """
    
    def __init__(
        self,
        models: Dict[int, nn.Module],
        noise_schedulers: Dict[int, NoiseScheduler],
        device: str = "cuda",
        prediction_type: str = "epsilon",
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50
    ):
        """
        Initialize progressive diffusion pipeline.
        
        Args:
            models: Dictionary mapping resolutions to models
            noise_schedulers: Dictionary mapping resolutions to schedulers
        """
        # Initialize with the highest resolution model
        max_res = max(models.keys())
        super().__init__(models[max_res], noise_schedulers[max_res], device, prediction_type, guidance_scale, num_inference_steps)
        
        self.models = {res: model.to(device) for res, model in models.items()}
        self.noise_schedulers = noise_schedulers
        self.resolutions = sorted(models.keys())
    
    def sample_progressive(
        self,
        shape: Tuple[int, ...],
        conditioning: Optional[torch.Tensor] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """
        Generate samples progressively from low to high resolution.
        
        Args:
            shape: Final shape of samples to generate
            conditioning: Optional conditioning information
            num_inference_steps: Number of denoising steps per resolution
            guidance_scale: Scale for classifier-free guidance
            generator: Random number generator
            
        Returns:
            High-resolution generated samples
        """
        if num_inference_steps is None:
            num_inference_steps = self.num_inference_steps
        if guidance_scale is None:
            guidance_scale = self.guidance_scale
        
        sample = None
        
        for resolution in self.resolutions:
            # Set current model and scheduler
            self.model = self.models[resolution]
            self.noise_scheduler = self.noise_schedulers[resolution]
            
            # Determine current shape
            current_shape = list(shape)
            scale_factor = resolution / max(self.resolutions)
            for i in range(1, 4):  # Spatial dimensions
                current_shape[i] = int(current_shape[i] * scale_factor)
            current_shape = tuple(current_shape)
            
            if sample is None:
                # Generate initial sample at lowest resolution
                sample = self.sample(
                    current_shape, conditioning, num_inference_steps, guidance_scale, generator
                )
            else:
                # Upsample previous sample
                sample = F.interpolate(
                    sample.permute(0, 4, 1, 2, 3),  # (B, D, H, W, C) -> (B, C, D, H, W)
                    size=current_shape[1:4],
                    mode='trilinear',
                    align_corners=False
                ).permute(0, 2, 3, 4, 1)  # (B, C, D, H, W) -> (B, D, H, W, C)
                
                # Refine with current resolution model
                sample = self.sample(
                    current_shape, conditioning, num_inference_steps // 2, guidance_scale, generator
                )
        
        return sample