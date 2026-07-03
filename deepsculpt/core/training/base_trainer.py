"""
Base trainer class for DeepSculpt PyTorch implementation.

This module provides the foundational trainer class that all specific trainers
inherit from, providing common functionality for training, validation, checkpointing,
and experiment tracking.
"""

import json
import os
import time
import logging
from typing import Dict, Any, Optional, Tuple, List, Union, Callable
from dataclasses import dataclass
from pathlib import Path
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

import numpy as np
from datetime import datetime

# TensorBoard disabled due to compatibility issues
TENSORBOARD_AVAILABLE = False
SummaryWriter = None

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    # Basic training parameters
    batch_size: int = 32
    learning_rate: float = 0.0002
    epochs: int = 100
    
    # Optimizer parameters
    beta1: float = 0.5
    beta2: float = 0.999
    weight_decay: float = 0.0
    
    # Training techniques
    mixed_precision: bool = True
    gradient_clip: float = 1.0
    progressive_growing: bool = False
    curriculum_learning: bool = False
    use_ema: bool = True
    ema_decay: float = 0.999
    r1_gamma: float = 10.0
    r1_interval: int = 16
    augment: str = "none"
    augment_p: float = 0.0
    augment_target: float = 0.6
    sample_from_ema: bool = True
    nan_guard: bool = True
    ttur_ratio: float = 0.25
    gan_loss_type: str = "wgan-gp"  # "softplus" or "wgan-gp"
    feature_matching_weight: float = 1.0
    use_disc_ema: bool = True
    disc_ema_decay: float = 0.999
    occupancy_loss_weight: float = 5.0
    occupancy_floor: float = 0.01
    occupancy_target_mode: str = "batch_real"
    dataset_occupancy_mean: Optional[float] = None
    dataset_occupancy_p10: Optional[float] = None
    dataset_occupancy_p90: Optional[float] = None
    
    # Distributed training
    distributed: bool = False
    world_size: int = 1
    rank: int = 0
    
    # Checkpointing and logging
    checkpoint_freq: int = 5
    log_freq: int = 250
    snapshot_freq: int = 1
    
    # Paths
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    snapshot_dir: str = "./snapshots"
    
    # Experiment tracking
    use_wandb: bool = False
    use_mlflow: bool = False
    use_tensorboard: bool = True
    experiment_name: str = "deepsculpt_experiment"
    
    # Early stopping
    early_stopping: bool = False
    patience: int = 10
    min_delta: float = 1e-4


class BaseTrainer(ABC):
    """
    Base class for all trainers with common functionality.
    
    Provides infrastructure for training, validation, checkpointing,
    experiment tracking, and distributed training.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: TrainingConfig,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda"
    ):
        """
        Initialize base trainer.
        
        Args:
            model: Model to train
            optimizer: Optimizer for training
            config: Training configuration
            scheduler: Learning rate scheduler
            device: Device for training
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        
        # Mixed precision training
        try:
            # New PyTorch 2.9+ API
            from torch.amp import GradScaler
            self.scaler = GradScaler('cuda') if config.mixed_precision and device != 'cpu' else None
        except ImportError:
            # Fallback for older PyTorch versions
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler() if config.mixed_precision else None
        
        # Metrics tracking
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'epoch_times': [],
            'learning_rates': []
        }
        
        # Current training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # Setup logging and experiment tracking
        self._setup_logging()
        self._setup_experiment_tracking()
        
        # Setup distributed training if enabled
        if config.distributed:
            self._setup_distributed()
        
        # Move model to device
        self.model = self.model.to(device)
    
    def _setup_logging(self):
        """Setup logging infrastructure."""
        os.makedirs(self.config.log_dir, exist_ok=True)
        
        # Setup Python logging
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(os.path.join(self.config.log_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Setup TensorBoard
        if self.config.use_tensorboard and TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(log_dir=self.config.log_dir)
        else:
            if self.config.use_tensorboard and not TENSORBOARD_AVAILABLE:
                print("Warning: TensorBoard requested but not available")
            self.writer = None
    
    def _setup_experiment_tracking(self):
        """Setup experiment tracking (Wandb, MLflow)."""
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project="deepsculpt",
                name=self.config.experiment_name,
                config=self.config.__dict__
            )
        
        if self.config.use_mlflow and MLFLOW_AVAILABLE:
            mlflow.set_experiment(self.config.experiment_name)
            mlflow.start_run()
            mlflow.log_params(self.config.__dict__)
    
    def _setup_distributed(self):
        """Setup distributed training."""
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')
        
        self.model = DDP(self.model, device_ids=[self.config.rank])
        self.logger.info(f"Distributed training setup complete. Rank: {self.config.rank}")
    
    @abstractmethod
    def train_step(self, batch: Any) -> Dict[str, float]:
        """
        Execute a single training step.
        
        Args:
            batch: Batch of training data
            
        Returns:
            Dictionary of step metrics
        """
        pass
    
    @abstractmethod
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data loader
            
        Returns:
            Dictionary of epoch metrics
        """
        pass
    
    @abstractmethod
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        """
        pass
    
    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        start_epoch: int = 0
    ) -> Dict[str, List[float]]:
        """
        Complete training loop.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            start_epoch: Starting epoch (for resuming training)
            
        Returns:
            Dictionary of training history
        """
        self.logger.info(f"Starting training for {self.config.epochs} epochs")
        
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'epoch_times': [],
            'learning_rates': []
        }
        
        for epoch in range(start_epoch, self.config.epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Training phase
            self.logger.info(f"Epoch {epoch + 1}/{self.config.epochs} - Training")
            train_metrics = self.train_epoch(train_dataloader)
            
            # Validation phase
            val_metrics = {}
            if val_dataloader is not None:
                self.logger.info(f"Epoch {epoch + 1}/{self.config.epochs} - Validation")
                val_metrics = self.validate(val_dataloader)
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Update metrics
            training_history['train_loss'].append(self._resolve_primary_loss(train_metrics))
            training_history['val_loss'].append(self._resolve_primary_loss(val_metrics) if val_metrics else 0)
            training_history['epoch_times'].append(epoch_time)
            training_history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # Log epoch metrics
            epoch_metrics = {**train_metrics}
            if val_metrics:
                epoch_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
            epoch_metrics['epoch_time'] = epoch_time
            epoch_metrics['learning_rate'] = self.optimizer.param_groups[0]['lr']
            
            self.log_metrics(epoch_metrics, epoch, "epoch")

            # Durable per-epoch metrics: cloud logs never see trainer INFO
            # lines (root logger is configured by the CLI before basicConfig
            # runs), so this JSONL — synced to GCS — is the only reliable
            # remote learning signal.
            try:
                os.makedirs(self.config.log_dir, exist_ok=True)
                with open(os.path.join(self.config.log_dir, "epoch_metrics.jsonl"), "a") as f:
                    f.write(json.dumps({"epoch": epoch + 1, **{k: float(v) for k, v in epoch_metrics.items() if isinstance(v, (int, float))}}) + "\n")
            except Exception:
                pass
            
            # Log to console
            train_loss = self._resolve_primary_loss(train_metrics)
            val_loss = self._resolve_primary_loss(val_metrics) if val_metrics else 0
            self.logger.info(
                f"Epoch {epoch + 1}/{self.config.epochs} completed in {epoch_time:.2f}s - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )
            
            # Check for improvement
            current_loss = val_loss if val_metrics else train_loss
            is_best = current_loss < self.best_loss
            if is_best:
                self.best_loss = current_loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            # Save checkpoint
            if (epoch + 1) % self.config.checkpoint_freq == 0 or is_best:
                checkpoint_path = os.path.join(
                    self.config.checkpoint_dir,
                    f"checkpoint_epoch_{epoch + 1}.pth"
                )
                self.save_checkpoint(checkpoint_path, epoch, epoch_metrics, is_best)

            self._after_epoch(epoch, train_metrics, val_metrics, is_best)
            
            # Early stopping
            if self.config.early_stopping and self.epochs_without_improvement >= self.config.patience:
                self.logger.info(f"Early stopping triggered after {self.config.patience} epochs without improvement")
                break
        
        self.logger.info("Training completed")
        return training_history

    def _after_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]],
        is_best: bool,
    ) -> None:
        """Optional per-epoch hook for trainer-specific snapshotting or reporting."""
        return None

    def _resolve_primary_loss(self, metrics: Optional[Dict[str, Any]]) -> float:
        """Resolve a primary loss value for logging/history across trainer types."""
        if not metrics:
            return 0.0

        for key in ("loss", "train_loss", "gen_loss", "diffusion_loss", "val_loss"):
            value = metrics.get(key)
            if isinstance(value, (int, float)):
                return float(value)

        return 0.0
    
    def save_checkpoint(self, path: str, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """
        Save training checkpoint.
        
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
            'epochs_without_improvement': self.epochs_without_improvement
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = path.replace('.pth', '_best.pth')
            torch.save(checkpoint, best_path)
        
        self.logger.info(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """
        Load training checkpoint.
        
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
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)
        
        self.logger.info(f"Checkpoint loaded: {path}")
        return checkpoint
    
    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        """
        Log metrics to all configured tracking systems.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Current step/epoch
            prefix: Prefix for metric names
        """
        # TensorBoard
        if self.writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"{prefix}/{key}" if prefix else key, value, step)
        
        # Wandb
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    wandb_metrics[f"{prefix}/{key}" if prefix else key] = value
            if wandb_metrics:
                wandb.log(wandb_metrics, step=step)
        
        # MLflow
        if self.config.use_mlflow and MLFLOW_AVAILABLE:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"{prefix}_{key}" if prefix else key, value, step=step)
    
    def get_training_info(self) -> Dict[str, Any]:
        """Get comprehensive training information."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "model_class": self.model.__class__.__name__,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "current_epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "epochs_without_improvement": self.epochs_without_improvement,
            "device": str(self.device),
            "distributed": self.config.distributed,
            "mixed_precision": self.config.mixed_precision,
            "learning_rate": self.optimizer.param_groups[0]['lr'],
        }
    
    def cleanup(self):
        """Cleanup resources."""
        if self.writer:
            self.writer.close()
        
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.finish()
        
        if self.config.use_mlflow and MLFLOW_AVAILABLE:
            mlflow.end_run()
        
        if self.config.distributed:
            dist.destroy_process_group()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
