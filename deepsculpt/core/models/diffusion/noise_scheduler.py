"""
Noise scheduling algorithms for diffusion models in DeepSculpt.

This module provides various noise scheduling strategies for the diffusion process,
including linear, cosine, and custom scheduling methods.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List, Union
import math
import numpy as np


class NoiseScheduler:
    """
    Base class for noise scheduling in diffusion models.
    
    Handles noise scheduling for the forward and reverse diffusion processes,
    providing methods for adding noise and computing various diffusion parameters.
    """
    
    def __init__(
        self,
        schedule_type: str = "linear",
        timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        device: str = "cuda"
    ):
        """
        Initialize the noise scheduler.
        
        Args:
            schedule_type: Type of noise schedule ("linear", "cosine", "sigmoid")
            timesteps: Number of diffusion timesteps
            beta_start: Starting value for beta schedule
            beta_end: Ending value for beta schedule
            device: Device to store tensors on
        """
        self.schedule_type = schedule_type
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device
        
        # Create noise schedule
        self.betas = self._create_beta_schedule()
        
        # Pre-compute useful values
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        
        # Calculations for reverse process
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # Move all tensors to device
        self._to_device()
    
    def _create_beta_schedule(self) -> torch.Tensor:
        """Create the beta schedule based on the specified type."""
        if self.schedule_type == "linear":
            return torch.linspace(self.beta_start, self.beta_end, self.timesteps)
        elif self.schedule_type == "cosine":
            return self._cosine_beta_schedule()
        elif self.schedule_type == "sigmoid":
            return self._sigmoid_beta_schedule()
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")
    
    def _cosine_beta_schedule(self) -> torch.Tensor:
        """Create cosine beta schedule."""
        s = 0.008
        steps = self.timesteps + 1
        x = torch.linspace(0, self.timesteps, steps)
        alphas_cumprod = torch.cos(((x / self.timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
    
    def _sigmoid_beta_schedule(self) -> torch.Tensor:
        """Create sigmoid beta schedule."""
        betas = torch.linspace(-6, 6, self.timesteps)
        betas = torch.sigmoid(betas) * (self.beta_end - self.beta_start) + self.beta_start
        return betas
    
    def _to_device(self):
        """Move all tensors to the specified device."""
        self.betas = self.betas.to(self.device)
        self.alphas = self.alphas.to(self.device)
        self.alphas_cumprod = self.alphas_cumprod.to(self.device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(self.device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(self.device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(self.device)
        self.posterior_variance = self.posterior_variance.to(self.device)
        self.sqrt_recip_alphas_cumprod = self.sqrt_recip_alphas_cumprod.to(self.device)
        self.sqrt_recipm1_alphas_cumprod = self.sqrt_recipm1_alphas_cumprod.to(self.device)
    
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Add noise to samples according to the noise schedule.
        
        Args:
            original_samples: Original clean samples
            noise: Random noise to add
            timesteps: Timesteps for each sample in the batch
            
        Returns:
            Noisy samples
        """
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps]
        
        # Reshape for broadcasting
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        noisy_samples = (
            sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        )
        
        return noisy_samples
    
    def get_velocity(
        self,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Get velocity for v-parameterization.
        
        Args:
            sample: Original sample
            noise: Noise tensor
            timesteps: Timesteps for each sample
            
        Returns:
            Velocity tensor
        """
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps]
        
        # Reshape for broadcasting
        while len(sqrt_alpha_prod.shape) < len(sample.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        return velocity
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        prediction_type: str = "epsilon"
    ) -> torch.Tensor:
        """
        Predict the sample at the previous timestep.
        
        Args:
            model_output: Output from the diffusion model
            timestep: Current timestep
            sample: Current sample
            prediction_type: Type of prediction ("epsilon", "sample", "v_prediction")
            
        Returns:
            Previous sample
        """
        t = timestep
        
        if prediction_type == "epsilon":
            # Predict x_0 from epsilon
            pred_original_sample = (
                sample - self.sqrt_one_minus_alphas_cumprod[t] * model_output
            ) / self.sqrt_alphas_cumprod[t]
        elif prediction_type == "sample":
            pred_original_sample = model_output
        elif prediction_type == "v_prediction":
            pred_original_sample = (
                self.sqrt_alphas_cumprod[t] * sample - 
                self.sqrt_one_minus_alphas_cumprod[t] * model_output
            )
        else:
            raise ValueError(f"Unknown prediction type: {prediction_type}")
        
        # Clip predicted x_0
        pred_original_sample = torch.clamp(pred_original_sample, -1, 1)
        
        # Compute coefficients for pred_original_sample and current sample
        pred_original_sample_coeff = (
            self.sqrt_alphas_cumprod_prev[t] * self.betas[t] / (1 - self.alphas_cumprod[t])
        )
        current_sample_coeff = (
            self.alphas[t] ** 0.5 * (1 - self.alphas_cumprod_prev[t]) / (1 - self.alphas_cumprod[t])
        )
        
        # Compute predicted previous sample
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample
        
        return pred_prev_sample
    
    def add_noise_to_timestep(self, sample: torch.Tensor, timestep: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise to a sample at a specific timestep.
        
        Args:
            sample: Clean sample
            timestep: Timestep to add noise for
            
        Returns:
            Tuple of (noisy_sample, noise)
        """
        noise = torch.randn_like(sample)
        timesteps = torch.full((sample.shape[0],), timestep, device=self.device, dtype=torch.long)
        noisy_sample = self.add_noise(sample, noise, timesteps)
        return noisy_sample, noise
    
    def get_schedule_info(self) -> Dict[str, Any]:
        """Get information about the noise schedule."""
        return {
            "schedule_type": self.schedule_type,
            "timesteps": self.timesteps,
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
            "device": str(self.device),
            "beta_min": self.betas.min().item(),
            "beta_max": self.betas.max().item(),
            "alpha_cumprod_min": self.alphas_cumprod.min().item(),
            "alpha_cumprod_max": self.alphas_cumprod.max().item(),
        }


class DDIMScheduler(NoiseScheduler):
    """
    DDIM (Denoising Diffusion Implicit Models) scheduler for faster sampling.
    
    Allows for deterministic sampling with fewer steps than the original DDPM.
    """
    
    def __init__(
        self,
        schedule_type: str = "linear",
        timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        device: str = "cuda",
        eta: float = 0.0,
        clip_sample: bool = True
    ):
        """
        Initialize DDIM scheduler.
        
        Args:
            eta: Parameter controlling stochasticity (0.0 = deterministic)
            clip_sample: Whether to clip predicted samples
        """
        super().__init__(schedule_type, timesteps, beta_start, beta_end, device)
        self.eta = eta
        self.clip_sample = clip_sample
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        prediction_type: str = "epsilon",
        eta: Optional[float] = None,
        prev_timestep: Optional[int] = None
    ) -> torch.Tensor:
        """
        DDIM sampling step.

        Args:
            model_output: Output from the diffusion model
            timestep: Current timestep
            sample: Current sample
            prediction_type: Type of prediction
            eta: Override eta value for this step
            prev_timestep: Next scheduled timestep when sampling with strided
                steps (e.g. 50 inference steps over 1000 trained). Defaults to
                timestep - 1 (unstrided).

        Returns:
            Previous sample
        """
        if eta is None:
            eta = self.eta

        t = timestep
        prev_t = prev_timestep if prev_timestep is not None else t - 1

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0)

        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        if prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        elif prediction_type == "sample":
            pred_original_sample = model_output
        else:
            raise ValueError(f"Unsupported prediction type: {prediction_type}")

        if self.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

        # Compute variance
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        std_dev_t = eta * variance ** 0.5

        # Direction term uses the epsilon implied by the (possibly clipped)
        # pred_original_sample, not the raw model output — correct for both
        # epsilon and sample prediction after clipping.
        pred_epsilon = (sample - alpha_prod_t ** 0.5 * pred_original_sample) / beta_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** 0.5 * pred_epsilon
        pred_prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        
        if eta > 0:
            noise = torch.randn_like(sample)
            pred_prev_sample = pred_prev_sample + std_dev_t * noise
        
        return pred_prev_sample


class DPMSolverScheduler(NoiseScheduler):
    """
    DPM-Solver scheduler for fast high-quality sampling.
    
    Implements the DPM-Solver algorithm for efficient diffusion model sampling.
    """
    
    def __init__(
        self,
        schedule_type: str = "linear",
        timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        device: str = "cuda",
        solver_order: int = 2,
        prediction_type: str = "epsilon"
    ):
        """
        Initialize DPM-Solver scheduler.
        
        Args:
            solver_order: Order of the DPM-Solver (1, 2, or 3)
            prediction_type: Type of model prediction
        """
        super().__init__(schedule_type, timesteps, beta_start, beta_end, device)
        self.solver_order = solver_order
        self.prediction_type = prediction_type
        
        # Pre-compute lambda values for DPM-Solver
        self.lambda_t = torch.log(self.alphas_cumprod) - torch.log(1 - self.alphas_cumprod)
    
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        prev_timestep: Optional[int] = None
    ) -> torch.Tensor:
        """
        DPM-Solver sampling step.
        
        Args:
            model_output: Output from the diffusion model
            timestep: Current timestep
            sample: Current sample
            prev_timestep: Previous timestep
            
        Returns:
            Previous sample
        """
        if prev_timestep is None:
            prev_timestep = timestep - 1
        
        lambda_t = self.lambda_t[timestep]
        lambda_s = self.lambda_t[prev_timestep] if prev_timestep >= 0 else torch.tensor(float('inf'))
        
        alpha_t = self.alphas_cumprod[timestep] ** 0.5
        alpha_s = self.alphas_cumprod[prev_timestep] ** 0.5 if prev_timestep >= 0 else torch.tensor(1.0)
        
        sigma_t = (1 - self.alphas_cumprod[timestep]) ** 0.5
        sigma_s = (1 - self.alphas_cumprod[prev_timestep]) ** 0.5 if prev_timestep >= 0 else torch.tensor(0.0)
        
        h = lambda_s - lambda_t
        
        if self.prediction_type == "epsilon":
            x_0_pred = (sample - sigma_t * model_output) / alpha_t
        elif self.prediction_type == "sample":
            x_0_pred = model_output
        else:
            raise ValueError(f"Unsupported prediction type: {self.prediction_type}")
        
        # First-order update
        if self.solver_order == 1:
            x_prev = alpha_s * x_0_pred + sigma_s * model_output
        else:
            # Higher-order updates would require storing previous model outputs
            # For simplicity, using first-order here
            x_prev = alpha_s * x_0_pred + sigma_s * model_output
        
        return x_prev


class AdaptiveScheduler(NoiseScheduler):
    """
    Adaptive noise scheduler that adjusts based on training progress.
    """
    
    def __init__(
        self,
        schedule_type: str = "linear",
        timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        device: str = "cuda",
        adaptation_rate: float = 0.01
    ):
        """
        Initialize adaptive scheduler.
        
        Args:
            adaptation_rate: Rate of adaptation based on loss
        """
        super().__init__(schedule_type, timesteps, beta_start, beta_end, device)
        self.adaptation_rate = adaptation_rate
        self.loss_history = []
    
    def update_schedule(self, loss: float):
        """
        Update the noise schedule based on training loss.
        
        Args:
            loss: Current training loss
        """
        self.loss_history.append(loss)
        
        # Adapt schedule if we have enough history
        if len(self.loss_history) > 10:
            recent_loss = np.mean(self.loss_history[-10:])
            older_loss = np.mean(self.loss_history[-20:-10]) if len(self.loss_history) > 20 else recent_loss
            
            # If loss is increasing, make schedule more aggressive
            if recent_loss > older_loss:
                self.beta_end = min(0.05, self.beta_end * (1 + self.adaptation_rate))
            # If loss is decreasing, make schedule more conservative
            else:
                self.beta_end = max(0.01, self.beta_end * (1 - self.adaptation_rate))
            
            # Recreate schedule with new parameters
            self.betas = self._create_beta_schedule()
            self._to_device()
    
    def get_adaptation_info(self) -> Dict[str, Any]:
        """Get information about schedule adaptation."""
        return {
            "current_beta_end": self.beta_end,
            "loss_history_length": len(self.loss_history),
            "recent_avg_loss": np.mean(self.loss_history[-10:]) if len(self.loss_history) >= 10 else None,
            "adaptation_rate": self.adaptation_rate,
        }