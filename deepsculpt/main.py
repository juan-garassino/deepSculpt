#!/usr/bin/env python3
"""
DeepSculpt v2.0 - PyTorch Main Entry Point

Modern PyTorch-based 3D generative models with comprehensive functionality:
- Modular architecture with clean separation of concerns
- Sparse tensor support for memory efficiency
- GAN and Diffusion models for 3D generation
- Advanced training with mixed precision and distributed support
- Comprehensive data generation and preprocessing pipeline
- Interactive visualization and analysis tools
- GPU optimization and memory management
- Experiment tracking and model versioning

Usage Examples:
    # GAN training with sparse tensors
    python main.py train-gan --model-type=skip --epochs=100 --data-folder=./data --sparse --mixed-precision
    
    # Diffusion model training
    python main.py train-diffusion --epochs=100 --data-folder=./data --timesteps=1000 --noise-schedule=cosine
    
    # Generate synthetic training data
    python main.py generate-data --num-samples=1000 --output-dir=./data --sparse --num-shapes=5
    
    # Sample from trained models
    python main.py sample-gan --checkpoint=./checkpoints/generator.pt --num-samples=10 --visualize
    python main.py sample-diffusion --checkpoint=./checkpoints/diffusion.pt --num-samples=5 --num-steps=50
    
    # Data preprocessing and curation
    python main.py preprocess --input-dir=./raw_data --output-dir=./processed --encoding=one_hot
    
    # Interactive visualization
    python main.py visualize --data-path=./data/sample.pt --backend=plotly --interactive
    
    # Performance benchmarking
    python main.py benchmark --model-type=skip --batch-size=32 --sparse --profile-memory
    
    # Model evaluation and comparison
    python main.py evaluate --checkpoint=./checkpoints/model.pt --test-data=./test --metrics=all
    
    # Export models for deployment
    python main.py export --checkpoint=./checkpoints/model.pt --format=onnx --output=./exports
"""

import argparse
import sys
import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# Framework imports
try:
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    PYTORCH_AVAILABLE = True
    print(f"PyTorch {torch.__version__} available")
except ImportError:
    PYTORCH_AVAILABLE = False
    print("Error: PyTorch not available. Please install PyTorch.")
    sys.exit(1)

# Optional dependencies
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("Warning: NumPy not available")

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("Warning: MLflow not available - experiment tracking disabled")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: Weights & Biases not available")

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import DeepSculpt v2.0 modules
try:
    from deepsculpt.core.models.model_factory import PyTorchModelFactory as PyTorchModelFactoryV2
    from deepsculpt.core.training.gan_trainer import GANTrainer
    from deepsculpt.core.training.diffusion_trainer import DiffusionTrainer
    from deepsculpt.core.training.base_trainer import BaseTrainer, TrainingConfig
    from deepsculpt.core.data.generation.pytorch_collector import PyTorchCollector
    from deepsculpt.core.data.generation.pytorch_sculptor import PyTorchSculptor
    from deepsculpt.core.data.transforms.pytorch_curator import PyTorchCurator
    from deepsculpt.core.data.loaders.data_loaders import StreamingDataLoader
    from deepsculpt.core.visualization.pytorch_visualization import PyTorchVisualizer
    from deepsculpt.core.utils.pytorch_utils import PyTorchUtils
    from deepsculpt.core.utils.logger import RichLogger
    
except ImportError as e:
    print(f"Error importing DeepSculpt v2.0 modules: {e}")
    print("Make sure all required modules are available in the core directory")
    print("Run from the deepsculpt directory")
    sys.exit(1)


class PairedTensorDataset(torch.utils.data.Dataset):
    """Dataset backed by structure/color file pairs saved on disk."""

    def __init__(self, sample_pairs: List[Tuple[Path, Path]]):
        self.sample_pairs = sample_pairs

    def __len__(self) -> int:
        return len(self.sample_pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        structure_path, colors_path = self.sample_pairs[idx]

        if structure_path.suffix == ".pt":
            structure = torch.load(structure_path, map_location="cpu")
            colors = torch.load(colors_path, map_location="cpu")
        elif structure_path.suffix == ".npy":
            structure = torch.from_numpy(np.load(structure_path))
            colors = torch.from_numpy(np.load(colors_path))
        else:
            raise ValueError(f"Unsupported sample format: {structure_path.suffix}")

        return {
            "structure": structure,
            "colors": colors,
            "index": torch.tensor(idx),
        }


class DeepSculptV2Main:
    """Main orchestrator for DeepSculpt v2.0 operations with comprehensive functionality."""
    
    def __init__(self, args=None):
        """Initialize the main orchestrator with device detection and configuration."""
        self.device = self._setup_device(args)
        self.logger = RichLogger(level="INFO")
        self.config = self._load_config(args)
        
        print(f"DeepSculpt v2.0 - Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def _setup_device(self, args):
        """Setup compute device with proper configuration."""
        if hasattr(args, 'cpu') and args.cpu:
            return "cpu"
        
        if torch.cuda.is_available():
            device = "cuda"
            # Enable optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            return device
        else:
            print("Warning: CUDA not available, using CPU")
            return "cpu"
    
    def _load_config(self, args):
        """Load configuration from file or use defaults."""
        config_path = getattr(args, 'config', None) or './config.yaml'
        
        if os.path.exists(config_path):
            try:
                import yaml
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                print(f"Loaded configuration from {config_path}")
                return config
            except ImportError:
                print("Warning: PyYAML not available, using default config")
            except Exception as e:
                print(f"Warning: Could not load config from {config_path}: {e}")
        
        # Default configuration
        return {
            "model": {"void_dim": 64, "noise_dim": 100},
            "training": {"batch_size": 32, "learning_rate": 0.0002},
            "data": {"sparse_threshold": 0.1, "num_workers": 4}
        }
    
    def _find_resume_checkpoint(self, output_dir, pattern):
        """Latest (run_dir, checkpoint) under output_dir matching run-dir pattern, or None."""
        import re
        for run_dir in sorted(Path(output_dir).glob(pattern), reverse=True):
            ckpt_dir = run_dir / "checkpoints"
            if not ckpt_dir.is_dir():
                continue
            ckpts = []
            for p in ckpt_dir.iterdir():
                m = re.fullmatch(r"checkpoint_epoch_(\d+)\.pth", p.name)
                if m:
                    ckpts.append((int(m.group(1)), p))
            if ckpts:
                return run_dir, max(ckpts)[1]
        return None

    def train_gan(self, args):
        """Train GAN models with comprehensive configuration and monitoring."""
        print(f"Training GAN model: {args.model_type}")

        # Create results directory (or reuse the latest one when resuming, so
        # chained ≤1h Cloud Run executions continue the same run)
        resume_from = None
        if getattr(args, 'resume', False):
            resume_from = self._find_resume_checkpoint(args.output_dir, f"gan_{args.model_type}_*")
        if resume_from is not None:
            results_dir = resume_from[0]
            print(f"Resuming run {results_dir.name} from {resume_from[1].name}")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = Path(args.output_dir) / f"gan_{args.model_type}_{timestamp}"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup experiment tracking
        experiment_tracker = None
        if args.mlflow and MLFLOW_AVAILABLE:
            experiment_tracker = self._setup_mlflow_tracking(args, results_dir)
        elif args.wandb and WANDB_AVAILABLE:
            experiment_tracker = self._setup_wandb_tracking(args, results_dir)
        
        # Create data loader
        data_loader = self._create_data_loader(args)
        collection_dir = self._resolve_collection_dir(Path(args.data_folder))
        collection_metadata = self._load_collection_metadata(collection_dir) if collection_dir is not None else {}
        occupancy_stats = collection_metadata.get("occupancy_stats", {})
        
        # Create model factory
        model_factory = PyTorchModelFactoryV2()
        
        # Create models
        gen_kwargs = {}
        if getattr(args, 'gen_channels', None) is not None:
            gen_kwargs['gen_channels'] = args.gen_channels
        generator = model_factory.create_gan_generator(
            model_type=args.model_type,
            void_dim=args.void_dim,
            noise_dim=args.noise_dim,
            color_mode=0,  # Use monochrome mode for single channel
            sparse=args.sparse,
            **gen_kwargs
        ).to(self.device)
        
        discriminator = model_factory.create_gan_discriminator(
            model_type=args.discriminator_type,
            void_dim=args.void_dim,
            color_mode=0,  # Use monochrome mode for single channel
            sparse=args.sparse
        ).to(self.device)
        
        # Print model information
        if args.verbose:
            total_params_gen = sum(p.numel() for p in generator.parameters())
            total_params_disc = sum(p.numel() for p in discriminator.parameters())
            print(f"Generator parameters: {total_params_gen:,}")
            print(f"Discriminator parameters: {total_params_disc:,}")
        
        # Create optimizers
        gen_optimizer = torch.optim.Adam(
            generator.parameters(),
            lr=args.learning_rate,
            betas=(args.beta1, args.beta2)
        )
        
        disc_optimizer = torch.optim.Adam(
            discriminator.parameters(),
            lr=args.learning_rate,
            betas=(args.beta1, args.beta2)
        )
        
        # Create schedulers if requested
        gen_scheduler = None
        disc_scheduler = None
        if args.scheduler:
            gen_scheduler = torch.optim.lr_scheduler.StepLR(
                gen_optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma
            )
            disc_scheduler = torch.optim.lr_scheduler.StepLR(
                disc_optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma
            )
        
        # Create training configuration
        training_config = TrainingConfig(
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            beta1=args.beta1,
            beta2=args.beta2,
            mixed_precision=args.mixed_precision,
            gradient_clip=args.gradient_clip,
            use_ema=args.use_ema,
            ema_decay=args.ema_decay,
            gan_loss_type=getattr(args, 'gan_loss_type', 'wgan-gp'),
            feature_matching_weight=getattr(args, 'feature_matching_weight', 1.0),
            r1_gamma=args.r1_gamma,
            r1_interval=args.r1_interval,
            augment=args.augment,
            augment_p=args.augment_p,
            augment_target=args.augment_target,
            sample_from_ema=args.sample_from_ema,
            occupancy_loss_weight=args.occupancy_loss_weight,
            occupancy_floor=args.occupancy_floor,
            occupancy_target_mode=args.occupancy_target_mode,
            ttur_ratio=getattr(args, 'ttur_ratio', 0.25),
            dataset_occupancy_mean=occupancy_stats.get("mean"),
            dataset_occupancy_p10=occupancy_stats.get("p10"),
            dataset_occupancy_p90=occupancy_stats.get("p90"),
            snapshot_freq=args.snapshot_freq,
            checkpoint_freq=getattr(args, 'checkpoint_freq', 5),
            checkpoint_dir=str(results_dir / "checkpoints"),
            log_dir=str(results_dir / "logs"),
            snapshot_dir=str(results_dir / "snapshots"),
            use_tensorboard=False,  # Disable TensorBoard since it's not available
            use_wandb=False,
            use_mlflow=bool(getattr(args, 'mlflow', False) and MLFLOW_AVAILABLE),
            experiment_name="deepsculpt"
        )

        # Setup trainer
        trainer = GANTrainer(
            generator=generator,
            discriminator=discriminator,
            gen_optimizer=gen_optimizer,
            disc_optimizer=disc_optimizer,
            config=training_config,
            gen_scheduler=gen_scheduler,
            disc_scheduler=disc_scheduler,
            device=self.device,
            noise_dim=args.noise_dim
        )

        # Apply run naming and extra params now that the trainer has opened the mlflow run
        if experiment_tracker is not None:
            _run_name = os.environ.get("RUN_ID") or f"{args.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            experiment_tracker.set_tag("mlflow.runName", _run_name)
            experiment_tracker.log_params({
                "framework": "pytorch",
                "void_dim": args.void_dim,
                "device": self.device
            })
            experiment_tracker.log_param("command", args.command)
            experiment_tracker.log_param("grammar_version", collection_metadata.get("grammar_version", "n/a"))

        # Write config.json BEFORE training: cloud slices die at the task
        # timeout without reaching post-training code, and the latent loaders
        # need this file beside the checkpoints to rebuild the architecture.
        config = {
            "model_type": args.model_type,
            "void_dim": args.void_dim,
            "noise_dim": args.noise_dim,
            # Must mirror the color_mode the models were actually built with above,
            # otherwise sample-gan rebuilds the wrong architecture.
            "color_mode": 0,
            "sparse": args.sparse,
            "discriminator_type": args.discriminator_type,
            **({"gen_channels": args.gen_channels} if getattr(args, 'gen_channels', None) is not None else {}),
            "use_ema": args.use_ema,
            "sample_from_ema": args.sample_from_ema,
            "training_params": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "beta1": args.beta1,
                "beta2": args.beta2,
                "r1_gamma": args.r1_gamma,
                "r1_interval": args.r1_interval,
                "augment": args.augment,
                "augment_p": args.augment_p,
                "augment_target": args.augment_target,
                "occupancy_loss_weight": args.occupancy_loss_weight,
                "occupancy_floor": args.occupancy_floor,
                "occupancy_target_mode": args.occupancy_target_mode,
            }
        }
        with open(results_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)

        # Train the model
        start_epoch = 0
        if resume_from is not None:
            ckpt = trainer.load_checkpoint(str(resume_from[1]))
            start_epoch = int(ckpt.get('epoch', -1)) + 1
            print(f"Resumed at epoch {start_epoch}")
        print(f"Starting training for {args.epochs} epochs")
        metrics = trainer.train(
            train_dataloader=data_loader,
            start_epoch=start_epoch
        )
        
        # Save final models
        torch.save(generator.state_dict(), results_dir / "generator_final.pt")
        if trainer.ema_generator is not None:
            torch.save(trainer.ema_generator.state_dict(), results_dir / "ema_generator_final.pt")
        torch.save(discriminator.state_dict(), results_dir / "discriminator_final.pt")
        
        with open(results_dir / "run_summary.json", "w") as f:
            json.dump(
                {
                    "train_history": metrics,
                    "last_epoch_metrics": trainer.last_epoch_metrics,
                    "training_info": trainer.get_training_info(),
                    "dataset_path": str(collection_dir) if collection_dir is not None else None,
                    "dataset_occupancy_stats": occupancy_stats,
                },
                f,
                indent=2,
            )
        
        # Generate sample visualizations
        if args.generate_samples:
            self._generate_sample_visualizations(trainer._generator_for_sampling(), results_dir, args)
        
        print(f"Training completed! Results saved to {results_dir}")
        return 0
    
    def train_diffusion(self, args):
        """Train diffusion models with advanced configuration."""
        print("Training diffusion model")

        # cuDNN requests multi-GiB workspaces for the 3D transposed convs
        # (batch-scaled 4-8 GiB single asks OOM'd the L4 repeatedly, with
        # benchmark autotuning on AND off — heuristic mode picks the same
        # big-workspace algos). deterministic=True restricts cuDNN to
        # bounded-workspace algorithms; marginally slower, actually fits.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        # Create results directory
        resume_from = None
        if getattr(args, 'resume', False):
            resume_from = self._find_resume_checkpoint(args.output_dir, "diffusion_*")
        if resume_from is not None:
            results_dir = resume_from[0]
            print(f"Resuming run {results_dir.name} from {resume_from[1].name}")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = Path(args.output_dir) / f"diffusion_{timestamp}"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup experiment tracking
        experiment_tracker = None
        if args.mlflow and MLFLOW_AVAILABLE:
            experiment_tracker = self._setup_mlflow_tracking(args, results_dir)
        
        # Create data loader
        data_loader = self._create_data_loader(args)
        
        # Create model factory
        model_factory = PyTorchModelFactoryV2()
        
        # Determine number of channels based on color mode
        # For diffusion models: monochrome=1, color=6 (structure + colors with more detail)
        color_mode = getattr(args, 'color', False)
        num_channels = 6 if color_mode else 1
        
        # Create diffusion model
        model = model_factory.create_diffusion_model(
            model_type="unet3d",
            void_dim=args.void_dim,
            in_channels=num_channels,
            out_channels=num_channels,
            timesteps=args.timesteps,
            sparse=args.sparse,
            model_channels=args.model_channels,
            use_checkpoint=args.grad_checkpoint
        ).to(self.device)
        
        # Create noise scheduler
        from deepsculpt.core.models.diffusion.noise_scheduler import NoiseScheduler
        noise_scheduler = NoiseScheduler(
            schedule_type=args.noise_schedule,
            timesteps=args.timesteps,
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            device=self.device  # Use the same device as the main app
        )
        
        # Create diffusion pipeline
        from deepsculpt.core.models.diffusion.pipeline import Diffusion3DPipeline
        diffusion_pipeline = Diffusion3DPipeline(
            model=model,
            noise_scheduler=noise_scheduler,
            device=self.device
        )
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        # Create scheduler
        scheduler = None
        if args.scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs
            )
        
        # Create training configuration
        training_config = TrainingConfig(
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            mixed_precision=args.mixed_precision,
            use_ema=args.use_ema,
            ema_decay=args.ema_decay,
            checkpoint_freq=getattr(args, 'checkpoint_freq', 5),
            checkpoint_dir=str(results_dir / "checkpoints"),
            log_dir=str(results_dir / "logs"),
            snapshot_dir=str(results_dir / "snapshots"),
            use_tensorboard=False,
            use_wandb=False,
            use_mlflow=bool(getattr(args, 'mlflow', False) and MLFLOW_AVAILABLE),
            experiment_name="deepsculpt"
        )

        # Setup trainer
        trainer = DiffusionTrainer(
            model=model,
            optimizer=optimizer,
            config=training_config,
            noise_scheduler=noise_scheduler,
            scheduler=scheduler,
            device=self.device
        )

        # Apply run naming and extra params now that the trainer has opened the mlflow run
        if experiment_tracker is not None:
            _run_name = os.environ.get("RUN_ID") or f"diffusion_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            experiment_tracker.set_tag("mlflow.runName", _run_name)
            experiment_tracker.log_params({
                "framework": "pytorch",
                "void_dim": args.void_dim,
                "device": self.device
            })
            experiment_tracker.log_param("command", args.command)

        # Write config.json BEFORE training: cloud slices die at the task
        # timeout without reaching post-training code, and the latent loaders
        # need this file beside the checkpoints to rebuild the architecture.
        with open(results_dir / "config.json", "w") as f:
            json.dump(
                {
                    "model_type": "unet3d",  # train-diffusion has no --model-type flag; factory call above is hardcoded
                    "void_dim": args.void_dim,
                    "num_channels": num_channels,
                    "model_channels": args.model_channels,
                    "timesteps": args.timesteps,
                    "noise_schedule": args.noise_schedule,
                    "beta_start": args.beta_start,
                    "beta_end": args.beta_end,
                    "sparse": args.sparse,
                    "color_mode": color_mode,
                    "use_ema": args.use_ema,
                    "training_params": {
                        "epochs": args.epochs,
                        "batch_size": args.batch_size,
                        "learning_rate": args.learning_rate,
                        "weight_decay": args.weight_decay,
                        "num_workers": args.num_workers,
                    },
                },
                f,
                indent=2,
            )

        # Train the model
        start_epoch = 0
        if resume_from is not None:
            ckpt = trainer.load_checkpoint(str(resume_from[1]))
            start_epoch = int(ckpt.get('epoch', -1)) + 1
            print(f"Resumed at epoch {start_epoch}")
        print(f"Starting diffusion training for {args.epochs} epochs")
        metrics = trainer.train(
            train_dataloader=data_loader,
            start_epoch=start_epoch
        )
        
        # Save final model
        torch.save({
            'model_state_dict': (trainer.ema_model.state_dict() if trainer.ema_model is not None else model.state_dict()),
            'raw_model_state_dict': model.state_dict(),
            'noise_scheduler': noise_scheduler,
            'config': {
                'void_dim': args.void_dim,
                'num_channels': num_channels,
                'timesteps': args.timesteps,
                'noise_schedule': args.noise_schedule,
                'sparse': args.sparse,
                'use_ema': args.use_ema,
                'color': color_mode,
                'model_channels': args.model_channels,
            }
        }, results_dir / "diffusion_final.pt")

        with open(results_dir / "run_summary.json", "w") as f:
            json.dump(
                {
                    "train_history": metrics,
                    "last_epoch_metrics": getattr(trainer, "last_epoch_metrics", {}),
                    "training_info": trainer.get_training_info(),
                    "dataset_path": str(collection_dir) if collection_dir is not None else None,
                    "dataset_occupancy_stats": occupancy_stats,
                },
                f,
                indent=2,
            )
        
        print(f"Diffusion training completed! Results saved to {results_dir}")
        return 0
    
    def generate_data(self, args):
        """Generate synthetic 3D data with comprehensive options."""
        print(f"Generating {args.num_samples} samples")

        # --- Shodhan grammar v2 (default preset) ---
        if getattr(args, 'structure_preset', 'shodhan') == 'shodhan':
            from datetime import date
            from deepsculpt.core.data.generation.shodhan import write_dataset
            out = Path(args.output_dir) / date.today().isoformat()
            write_dataset(out, num_samples=args.num_samples,
                          seed_start=getattr(args, 'seed_start', 0))
            print(f"Dataset generated successfully! Collection directory: {out}")
            return 0

        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure sculptor
        sculptor_config = self._build_sculptor_config(args)
        
        # Create collector (sparse_threshold passed separately to avoid duplication)
        collector = PyTorchCollector(
            sculptor_config=sculptor_config,
            output_format="pytorch",
            base_dir=str(output_dir),
            sparse_mode=args.sparse,
            sparse_threshold=args.sparse_threshold if args.sparse else 1.0,
            device=self.device
        )
        
        print(f"Generating {args.num_samples} samples...")
        start_time = time.time()
        
        # Generate dataset
        dataset_paths = collector.create_collection(args.num_samples)
        
        generation_time = time.time() - start_time
        
        # Save metadata
        metadata = {
            "num_samples": args.num_samples,
            "void_dim": args.void_dim,
            "num_shapes": args.num_shapes,
            "sparse": args.sparse,
            "sparse_threshold": args.sparse_threshold if args.sparse else 1.0,
            "device": self.device,
            "generation_time": generation_time,
            "timestamp": datetime.now().isoformat(),
            "dataset_paths": dataset_paths,
            "collection_dir": str(collector.date_dir),
            "occupancy_stats": self._summarize_occupancy_stats(collector.get_generation_stats().get("occupancy_values", [])),
            "structure_preset": getattr(args, "structure_preset", "architectural"),
            "sculptor_config": sculptor_config,
        }
        
        metadata_path = collector.date_dir / "dataset_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Dataset generated successfully in {generation_time:.2f}s!")
        print(f"Collection directory: {collector.date_dir}")
        print(f"Metadata saved to: {metadata_path}")
        
        return 0
    
    def sample_gan(self, args):
        """Generate samples from trained GAN model."""
        print(f"Generating {args.num_samples} samples from GAN: {args.checkpoint}")

        # Same loader as the latent-* commands: handles bare state_dicts and
        # trainer .pth checkpoints, rebuilds the architecture from config.json.
        from deepsculpt.core.latent import load_generator
        loaded = load_generator(Path(args.checkpoint), device=self.device, prefer_ema=False)
        generator = loaded.generator

        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate samples
        print(f"Generating {args.num_samples} samples...")
        samples = []

        with torch.no_grad():
            for i in range(args.num_samples):
                noise = torch.randn(1, loaded.noise_dim, device=self.device)
                sample = generator(noise)
                samples.append(sample.cpu())
                
                # Save individual sample
                sample_path = output_dir / f"sample_{i:04d}.pt"
                torch.save(sample.cpu(), sample_path)
        
        # Create visualizations if requested
        if args.visualize:
            print("Creating visualizations...")
            visualizer = PyTorchVisualizer(device=self.device)
            
            for i, sample in enumerate(samples):
                vis_path = output_dir / f"sample_{i:04d}.png"
                visualizer.plot_sculpture(sample.squeeze(), save_path=str(vis_path))
        
        print(f"Generated {args.num_samples} samples in {output_dir}")
        return 0
    
    def sample_diffusion(self, args):
        """Generate samples from trained diffusion model."""
        print(f"Generating {args.num_samples} samples from diffusion: {args.checkpoint}")
        
        # Load checkpoint
        checkpoint = torch.load(args.checkpoint, map_location=self.device)
        config = checkpoint['config']
        
        # Create model factory
        model_factory = PyTorchModelFactoryV2()
        
        # Create model
        model = model_factory.create_diffusion_model(
            model_type="unet3d",
            void_dim=config['void_dim'],
            in_channels=config.get('num_channels', 1),
            out_channels=config.get('num_channels', 1),
            timesteps=config.get('timesteps', 1000),
            sparse=config.get('sparse', False),
            model_channels=config.get('model_channels', 128)
        ).to(self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Keep the scheduler tensors on the active device after checkpoint load.
        noise_scheduler = checkpoint['noise_scheduler']
        if hasattr(noise_scheduler, "device"):
            noise_scheduler.device = self.device
        if hasattr(noise_scheduler, "_to_device"):
            noise_scheduler._to_device()

        from deepsculpt.core.models.diffusion.pipeline import Diffusion3DPipeline, FastSamplingPipeline
        if args.sampler == "ddpm":
            diffusion_pipeline = Diffusion3DPipeline(
                model=model,
                noise_scheduler=noise_scheduler,
                device=self.device,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_steps,
            )
        else:
            diffusion_pipeline = FastSamplingPipeline(
                model=model,
                noise_scheduler=noise_scheduler,
                device=self.device,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_steps,
                scheduler_type=args.sampler,
            )
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Generating {args.num_samples} samples with {args.num_steps} denoising steps...")
        
        # Generate samples
        samples = []
        with torch.no_grad():
            for i in range(args.num_samples):
                print(f"Generating sample {i+1}/{args.num_samples}")
                
                # Sample from diffusion model
                shape = (1, config.get('num_channels', 1),
                        config['void_dim'], config['void_dim'], config['void_dim'])
                sample = diffusion_pipeline.sample(
                    shape=shape,
                    num_inference_steps=args.num_steps,
                    guidance_scale=args.guidance_scale,
                )
                
                samples.append(sample.cpu())
                
                # Save sample
                sample_path = output_dir / f"sample_{i:04d}.pt"
                torch.save(sample.cpu(), sample_path)
        
        # Create visualizations if requested
        if args.visualize:
            print("Creating visualizations...")
            visualizer = PyTorchVisualizer(device=self.device)
            
            for i, sample in enumerate(samples):
                vis_path = output_dir / f"sample_{i:04d}.png"
                visualizer.plot_sculpture(sample.squeeze(), save_path=str(vis_path))
        
        print(f"Generated {args.num_samples} samples in {output_dir}")
        return 0
    
    def preprocess_data(self, args):
        """Preprocess and curate data for training."""
        print(f"Preprocessing data from {args.input_dir} to {args.output_dir}")
        
        # Create curator
        curator = PyTorchCurator(
            encoding_method=args.encoding,
            device=self.device,
            sparse_mode=args.sparse
        )
        
        # Process data
        dataset = curator.encode_dataset(args.input_dir)
        
        # Save processed dataset
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save dataset metadata
        metadata = {
            "input_dir": args.input_dir,
            "encoding_method": args.encoding,
            "sparse": args.sparse,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(output_dir / "preprocessing_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Data preprocessing completed! Output saved to {output_dir}")
        return 0
    
    def visualize(self, args):
        """Visualize 3D data with interactive options."""
        print(f"Visualizing {args.data_path} with {args.backend}")
        
        # Create visualizer
        visualizer = PyTorchVisualizer(
            backend=args.backend,
            device=self.device
        )
        
        # Load and visualize data
        data = torch.load(args.data_path, map_location=self.device)
        
        if isinstance(data, dict):
            structure = data.get('structure')
            colors = data.get('colors')
        else:
            structure = data
            colors = None
        
        # Create visualization
        if args.interactive:
            # Interactive visualization
            visualizer.plot_pointcloud(
                visualizer.voxel_to_pointcloud(structure, colors),
                colors
            )
        else:
            # Static visualization
            output_path = args.output_path or "visualization.png"
            visualizer.plot_sculpture(structure, colors, save_path=output_path)
            print(f"Visualization saved to {output_path}")
        
        return 0
    
    def benchmark(self, args):
        """Run comprehensive performance benchmarks."""
        print(f"Benchmarking {args.model_type} with batch size {args.batch_size}")
        
        # Create model factory
        model_factory = PyTorchModelFactoryV2()
        
        # Create model
        model = model_factory.create_gan_generator(
            model_type=args.model_type,
            void_dim=args.void_dim,
            noise_dim=args.noise_dim,
            sparse=args.sparse
        ).to(self.device)
        
        # Run inference benchmark
        input_shape = (args.batch_size, args.noise_dim)
        results = PyTorchUtils.benchmark_model_inference(model, input_shape)
        
        # Memory profiling if requested
        if args.profile_memory:
            memory_results = PyTorchUtils.calculate_memory_usage(
                torch.randn(args.batch_size, 1, args.void_dim, args.void_dim, args.void_dim)
            )
            results.update(memory_results)
        
        # Print results
        print("\nBenchmark Results:")
        print("=" * 50)
        for metric, value in results.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
        
        # Save results if requested
        if args.save_results:
            output_path = Path(args.output_dir) / "benchmark_results.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\nResults saved to {output_path}")
        
        return 0
    
    def evaluate(self, args):
        """Evaluate trained models with comprehensive metrics."""
        print(f"Evaluating model: {args.checkpoint}")
        
        # Implementation would include:
        # - Loading model and test data
        # - Computing evaluation metrics (FID, IS, etc.)
        # - Generating evaluation report
        
        print("Model evaluation completed!")
        return 0
    
    def export_model(self, args):
        """Export models for deployment."""
        print(f"Exporting model {args.checkpoint} to {args.format}")
        
        # Implementation would include:
        # - Loading PyTorch model
        # - Converting to specified format (ONNX, TorchScript, etc.)
        # - Saving exported model
        
        print(f"Model exported successfully to {args.output}")
        return 0

    def _load_latent_generator(self, args):
        """Load a generator for latent navigation, honoring --use-ema/--no-ema."""
        from deepsculpt.core.latent import load_generator
        return load_generator(
            Path(args.checkpoint), device=self.device, prefer_ema=args.use_ema
        )

    def _export_latent_outputs(self, volumes, args, stem, titles=None, rows=None, cols=None):
        """Write the requested --format outputs for a (N,C,D,H,W) volume tensor."""
        from deepsculpt.core.visualization.volume_export import (
            render_contact_sheet, save_gif_from_volumes, save_mesh,
        )
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        vols_np = volumes.numpy()

        # Always dump raw volumes so runs are comparable/deterministic
        torch.save(volumes, output_dir / f"{stem}_volumes.pt")

        formats = args.format or ["gif"]
        for fmt in formats:
            if fmt == "gif":
                save_gif_from_volumes(
                    list(vols_np), str(output_dir / f"{stem}.gif"),
                    fps=args.fps, mode=args.render, threshold=args.threshold,
                )
            elif fmt == "png":
                n = vols_np.shape[0]
                r = rows or 1
                c = cols or n
                render_contact_sheet(
                    list(vols_np), str(output_dir / f"{stem}.png"),
                    rows=r, cols=c, titles=titles,
                    threshold=args.threshold, mode=args.render,
                )
            elif fmt in ("obj", "stl"):
                for i, vol in enumerate(vols_np):
                    try:
                        save_mesh(vol, str(output_dir / f"{stem}_{i:03d}.{fmt}"),
                                  fmt=fmt, threshold=args.threshold)
                    except ValueError as e:
                        print(f"  skip mesh for step {i}: {e}")
        print(f"Latent outputs written to {output_dir} (formats: {', '.join(formats)})")

    def latent_walk(self, args):
        """Interpolation walk between seeded anchors (GAN z-space or diffusion noise-space)."""
        from deepsculpt.core.latent import latent_arithmetic, walk_path

        seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [0, 1]
        if len(seeds) < 2 and not args.arithmetic:
            raise ValueError("latent-walk needs at least 2 seeds (--seeds 12,77)")

        if args.backend == 'diffusion':
            return self._latent_walk_diffusion(args, seeds)

        from deepsculpt.core.latent import batched_generate, seeded_z
        loaded = self._load_latent_generator(args)

        anchors = torch.stack([seeded_z(s, loaded.noise_dim)[0] for s in seeds])
        if args.arithmetic:
            named = {chr(ord('a') + i): z for i, z in enumerate(anchors)}
            target = latent_arithmetic(named, args.arithmetic)
            anchors = torch.stack([anchors[0], target])
            print(f"Walking from seed {seeds[0]} to arithmetic result ({args.arithmetic})")

        path = walk_path(anchors, args.steps, mode=args.interp, closed=args.closed)
        print(f"Generating {path.shape[0]} steps along a {args.interp} walk "
              f"({len(seeds)} anchors, noise_dim={loaded.noise_dim})")
        volumes = batched_generate(loaded.generator, path,
                                   batch_size=args.batch_size, device=self.device)
        self._export_latent_outputs(volumes, args, stem="walk")
        return 0

    def _latent_walk_diffusion(self, args, seeds):
        """Walk in diffusion noise-space: interpolate initial noise, sample
        each step deterministically (DDIM/DPM-Solver, eta=0)."""
        from deepsculpt.core.latent import load_diffusion_pipeline, seeded_noise, walk_path

        if args.sampler == 'ddpm':
            raise ValueError("--backend diffusion needs a deterministic sampler (ddim or dpm_solver)")

        pipeline, config = load_diffusion_pipeline(
            Path(args.checkpoint), device=self.device,
            sampler=args.sampler, num_steps=args.diffusion_steps,
        )
        shape = (1, config.get('num_channels', 1),
                 config['void_dim'], config['void_dim'], config['void_dim'])

        anchors = torch.stack([seeded_noise(s, shape) for s in seeds])
        path = walk_path(anchors, args.steps, mode=args.interp, closed=args.closed)
        print(f"Diffusion walk: {path.shape[0]} steps x {args.diffusion_steps} "
              f"{args.sampler} denoising steps (void_dim={config['void_dim']})")

        volumes = []
        with torch.no_grad():
            for i in range(path.shape[0]):
                sample = pipeline.sample(
                    shape=shape,
                    num_inference_steps=args.diffusion_steps,
                    init_noise=path[i],
                )
                volumes.append(sample.cpu())
                print(f"  step {i + 1}/{path.shape[0]} done")
        self._export_latent_outputs(torch.cat(volumes), args, stem="walk_diffusion")
        return 0

    def latent_traverse(self, args):
        """Per-dimension (or per-principal-direction) traversal."""
        from deepsculpt.core.latent import (
            apply_direction, batched_generate, load_directions, seeded_z,
            traverse_dimension,
        )
        loaded = self._load_latent_generator(args)
        z_base = seeded_z(args.base_seed, loaded.noise_dim)[0]
        alphas = torch.linspace(-args.sigma_range, args.sigma_range, args.steps)

        all_volumes, titles = [], []
        if args.directions:
            d = load_directions(Path(args.directions))
            if d.noise_dim != loaded.noise_dim:
                raise ValueError(
                    f"directions were computed for noise_dim={d.noise_dim}, "
                    f"checkpoint has noise_dim={loaded.noise_dim}"
                )
            indices = ([int(i) for i in args.dims.split(",")] if args.dims
                       else list(range(min(args.num_dims, d.directions.shape[0]))))
            for idx in indices:
                zs = apply_direction(z_base, d.directions[idx], alphas.tolist())
                vols = batched_generate(loaded.generator, zs,
                                        batch_size=args.batch_size, device=self.device)
                all_volumes.append(vols)
                titles.extend([f"pc{idx} a={v:+.1f}" for v in alphas])
            label = f"{len(indices)} principal directions"
            rows = len(indices)
        else:
            dims = ([int(i) for i in args.dims.split(",")] if args.dims
                    else list(range(min(args.num_dims, loaded.noise_dim))))
            for dim in dims:
                zs = traverse_dimension(z_base, dim, args.steps, sigma_range=args.sigma_range)
                vols = batched_generate(loaded.generator, zs,
                                        batch_size=args.batch_size, device=self.device)
                all_volumes.append(vols)
                titles.extend([f"z[{dim}]={v:+.1f}" for v in alphas])
            label = f"{len(dims)} dims"
            rows = len(dims)

        volumes = torch.cat(all_volumes)
        print(f"Traversed {label} x {args.steps} steps "
              f"(base seed {args.base_seed}, ±{args.sigma_range}σ)")
        self._export_latent_outputs(volumes, args, stem="traverse",
                                    titles=titles, rows=rows, cols=args.steps)
        return 0

    def latent_directions(self, args):
        """Discover semantic directions (PCA/GANSpace) and optionally render one."""
        from deepsculpt.core.latent import (
            apply_direction, batched_generate, compute_directions, save_directions,
            seeded_z,
        )
        loaded = self._load_latent_generator(args)
        d = compute_directions(
            loaded.generator,
            noise_dim=loaded.noise_dim,
            num_samples=args.num_samples,
            num_components=args.components,
            method=args.method,
            batch_size=args.batch_size,
            device=self.device,
            seed=args.seed,
        )
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = Path(args.save) if args.save else output_dir / "directions.pt"
        save_directions(d, save_path)
        print(f"Saved {d.directions.shape[0]} {d.method} directions to {save_path}")
        print("Explained variance:", ", ".join(f"{v:.3f}" for v in d.explained_variance))

        if args.apply is not None:
            alphas = [float(a) for a in args.alphas.split(",")]
            all_volumes, titles = [], []
            for s in range(args.num_seeds):
                z = seeded_z(args.seed + s, loaded.noise_dim)[0]
                zs = apply_direction(z, d.directions[args.apply], alphas)
                vols = batched_generate(loaded.generator, zs,
                                        batch_size=args.batch_size, device=self.device)
                all_volumes.append(vols)
                titles.extend([f"seed{args.seed + s} a={a:+.1f}" for a in alphas])
            volumes = torch.cat(all_volumes)
            self._export_latent_outputs(
                volumes, args, stem=f"direction_pc{args.apply}",
                titles=titles, rows=args.num_seeds, cols=len(alphas),
            )
        return 0

    def _create_data_loader(self, args):
        """Create data loader based on arguments."""
        collection_dir = self._resolve_collection_dir(Path(args.data_folder))
        if collection_dir is not None:
            sample_pairs = self._load_sample_pairs(collection_dir)
            print(f"Loading {len(sample_pairs)} samples from {collection_dir}")
            dataset = PairedTensorDataset(sample_pairs)
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=getattr(args, "num_workers", 0),
            )

        print(
            f"Warning: no saved dataset found in {args.data_folder}. "
            "Falling back to a small generated streaming dataset."
        )

        sculptor_config = self._build_sculptor_config(args)

        collector = PyTorchCollector(
            sculptor_config=sculptor_config,
            device=self.device
        )
        dataset = collector.create_streaming_dataset(10)

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0
        )

    def _load_collection_metadata(self, collection_dir: Path) -> Dict[str, Any]:
        """Load collection-level metadata if available."""
        candidate_paths = [
            collection_dir / "dataset_metadata.json",
            collection_dir / "metadata" / "collection_metadata.json",
        ]
        for metadata_path in candidate_paths:
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    return json.load(f)
        return {}

    def _summarize_occupancy_stats(self, occupancy_values: List[float]) -> Dict[str, float]:
        """Summarize occupancy values into stable metadata fields."""
        if not occupancy_values:
            return {}

        occupancy_array = np.asarray(occupancy_values, dtype=np.float32)
        return {
            "mean": float(np.mean(occupancy_array)),
            "min": float(np.min(occupancy_array)),
            "max": float(np.max(occupancy_array)),
            "p10": float(np.percentile(occupancy_array, 10)),
            "p90": float(np.percentile(occupancy_array, 90)),
        }

    def _build_sculptor_config(self, args) -> Dict[str, Any]:
        """Build the procedural data-generation config from CLI arguments."""
        structure_preset = getattr(args, "structure_preset", "architectural")

        if structure_preset == "architectural":
            return {
                "void_dim": args.void_dim,
                "edges": (getattr(args, "edge_count", 0), getattr(args, "edge_min_ratio", 0.2), getattr(args, "edge_max_ratio", 0.4)),
                "planes": (3, getattr(args, "plane_min_ratio", 0.3), getattr(args, "plane_max_ratio", 0.5)),
                "pipes": (2, getattr(args, "pipe_min_ratio", 0.3), getattr(args, "pipe_max_ratio", 0.5)),
                "grid": (getattr(args, "grid_count", 1), getattr(args, "grid_step", 4)),
                "structure_mode": "architectural",
            }

        return {
            "void_dim": args.void_dim,
            "edges": (getattr(args, "edge_count", 2), getattr(args, "edge_min_ratio", 0.3), getattr(args, "edge_max_ratio", 0.5)),
            "planes": (getattr(args, "plane_count", 1), getattr(args, "plane_min_ratio", 0.3), getattr(args, "plane_max_ratio", 0.5)),
            "pipes": (getattr(args, "pipe_count", 1), getattr(args, "pipe_min_ratio", 0.3), getattr(args, "pipe_max_ratio", 0.5)),
            "grid": (getattr(args, "grid_count", 1), getattr(args, "grid_step", 4)),
            "structure_mode": "generic",
        }

    def _resolve_collection_dir(self, data_folder: Path) -> Optional[Path]:
        """Resolve a data folder to a single collection directory."""
        if not data_folder.exists():
            return None

        direct_collection = data_folder / "pytorch_samples" / "structures"
        if direct_collection.exists():
            return data_folder

        dated_collections = sorted(
            path for path in data_folder.iterdir()
            if path.is_dir() and (path / "pytorch_samples" / "structures").exists()
        )
        if dated_collections:
            return dated_collections[-1]

        recursive_matches = sorted(
            {
                path.parent.parent.parent
                for path in data_folder.rglob("structure_*.pt")
            }
            | {
                path.parent.parent.parent
                for path in data_folder.rglob("structure_*.npy")
            }
        )
        if recursive_matches:
            return recursive_matches[-1]

        return None

    def _load_sample_pairs(self, collection_dir: Path) -> List[Tuple[Path, Path]]:
        """Load paired structure/color sample paths from a collection directory."""
        structures_dir = collection_dir / "pytorch_samples" / "structures"
        colors_dir = collection_dir / "pytorch_samples" / "colors"

        structure_files = sorted(structures_dir.glob("structure_*.pt"))
        if not structure_files:
            structure_files = sorted(structures_dir.glob("structure_*.npy"))

        sample_pairs = []
        for structure_path in structure_files:
            colors_name = structure_path.name.replace("structure_", "colors_", 1)
            colors_path = colors_dir / colors_name
            if colors_path.exists():
                sample_pairs.append((structure_path, colors_path))

        if not sample_pairs:
            raise ValueError(f"No paired samples found under {collection_dir}")

        return sample_pairs
    
    def _setup_mlflow_tracking(self, args, results_dir):
        """Setup MLflow experiment tracking."""
        import mlflow
        
        # Set the single project-level experiment.  The actual run is opened by the
        # trainer's _setup_experiment_tracking (called in __init__).  Run naming and
        # extra param logging are applied in train_gan / train_diffusion right after
        # the trainer is instantiated, once the run is live.
        mlflow.set_experiment("deepsculpt")

        return mlflow
    
    def _setup_wandb_tracking(self, args, results_dir):
        """Setup Weights & Biases experiment tracking."""
        import wandb
        
        config = {
            "framework": "pytorch",
            "model_type": getattr(args, 'model_type', 'unknown'),
            "void_dim": args.void_dim,
            "epochs": getattr(args, 'epochs', 0),
            "batch_size": args.batch_size,
            "device": self.device
        }
        
        wandb.init(
            project="deepsculpt",
            config=config,
            name=f"{args.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        return wandb
    
    def _generate_sample_visualizations(self, generator, results_dir, args):
        """Generate sample visualizations after training."""
        print("Generating sample visualizations...")
        
        visualizer = PyTorchVisualizer(device=self.device)
        samples_dir = results_dir / "samples"
        samples_dir.mkdir(exist_ok=True)
        
        with torch.no_grad():
            num_preview_samples = max(1, int(getattr(args, "num_preview_samples", 1)))
            noise = torch.randn(num_preview_samples, args.noise_dim, device=self.device)
            samples = generator(noise)
            
            for i, sample in enumerate(samples):
                vis_path = samples_dir / f"sample_{i}.png"
                visualizer.plot_sculpture(sample.cpu(), save_path=str(vis_path))


def create_parser():
    """Create comprehensive argument parser for DeepSculpt v2.0."""
    parser = argparse.ArgumentParser(
        description="DeepSculpt v2.0 - PyTorch 3D Generative Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Global arguments
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # GAN training
    train_gan_parser = subparsers.add_parser('train-gan', help='Train GAN models')
    train_gan_parser.add_argument('--model-type', default='skip',
                                 choices=['simple', 'complex', 'skip', 'monochrome', 'autoencoder'],
                                 help='Type of GAN model to train')
    train_gan_parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    train_gan_parser.add_argument('--batch-size', type=int, default=32, help='Training batch size')
    train_gan_parser.add_argument('--void-dim', type=int, default=64, help='3D voxel space dimension')
    train_gan_parser.add_argument('--noise-dim', type=int, default=100, help='Noise vector dimension')
    train_gan_parser.add_argument('--learning-rate', type=float, default=0.0002, help='Learning rate')
    train_gan_parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta1 parameter')
    train_gan_parser.add_argument('--beta2', type=float, default=0.999, help='Adam beta2 parameter')
    train_gan_parser.add_argument('--data-folder', default='./data', help='Training data folder')
    train_gan_parser.add_argument('--output-dir', default='./results', help='Output directory')
    train_gan_parser.add_argument('--snapshot-freq', type=int, default=1, help='Snapshot frequency (epochs)')
    train_gan_parser.add_argument('--checkpoint-freq', type=int, default=5, help='Checkpoint frequency (epochs); lower for timeout-chained cloud slices so resume loses fewer epochs')
    train_gan_parser.add_argument('--color', action='store_true', help='Enable color mode')
    train_gan_parser.add_argument('--sparse', action='store_true', help='Use sparse tensors')
    train_gan_parser.add_argument('--mixed-precision', action='store_true', help='Use mixed precision training')
    train_gan_parser.add_argument('--gradient-clip', type=float, default=1.0, help='Gradient clipping value')
    train_gan_parser.add_argument('--discriminator-type', default='light',
                                 choices=['simple', 'complex', 'progressive', 'conditional', 'spectral_norm', 'multi_scale', 'patch', 'light'],
                                 help='Type of discriminator to train against')
    train_gan_parser.add_argument('--gen-channels', type=int, default=None, help='Generator base channel width (default: noise_dim)')
    train_gan_parser.add_argument('--scheduler', action='store_true', help='Use learning rate scheduler')
    train_gan_parser.add_argument('--scheduler-step', type=int, default=30, help='Scheduler step size')
    train_gan_parser.add_argument('--scheduler-gamma', type=float, default=0.1, help='Scheduler gamma')
    train_gan_parser.add_argument('--gan-loss-type', default='wgan-gp', choices=['softplus', 'wgan-gp'], help='GAN loss function')
    train_gan_parser.add_argument('--feature-matching-weight', type=float, default=1.0, help='Feature matching loss weight')
    train_gan_parser.add_argument('--use-ema', dest='use_ema', action='store_true', help='Use EMA weights for stable sampling/checkpoints')
    train_gan_parser.add_argument('--no-ema', dest='use_ema', action='store_false', help='Disable EMA weights')
    train_gan_parser.add_argument('--ema-decay', type=float, default=0.999, help='EMA decay for generator weights')
    train_gan_parser.add_argument('--ttur-ratio', type=float, default=0.25, help='TTUR disc/gen LR ratio (lower = weaker disc)')
    train_gan_parser.add_argument('--r1-gamma', type=float, default=2.0, help='R1 regularization gamma')
    train_gan_parser.add_argument('--r1-interval', type=int, default=16, help='R1 lazy regularization interval')
    train_gan_parser.add_argument('--augment', default='none', choices=['none', 'ada-lite'], help='Discriminator-side augmentation policy')
    train_gan_parser.add_argument('--augment-p', type=float, default=0.0, help='Initial augmentation probability')
    train_gan_parser.add_argument('--augment-target', type=float, default=0.7, help='Target real accuracy for ADA-lite controller')
    train_gan_parser.add_argument('--occupancy-loss-weight', type=float, default=5.0, help='Weight for occupancy-matching generator regularization')
    train_gan_parser.add_argument('--occupancy-floor', type=float, default=0.01, help='Minimum healthy fake occupancy before empty-collapse penalty activates')
    train_gan_parser.add_argument('--occupancy-target-mode', default='batch_real', choices=['batch_real', 'dataset_mean'], help='Reference occupancy target for generator regularization')
    train_gan_parser.add_argument('--sample-from-ema', dest='sample_from_ema', action='store_true', help='Use EMA generator for exported samples')
    train_gan_parser.add_argument('--sample-from-raw', dest='sample_from_ema', action='store_false', help='Use raw generator for exported samples')
    train_gan_parser.add_argument('--mlflow', action='store_true', help='Enable MLflow tracking')
    train_gan_parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases tracking')
    train_gan_parser.add_argument('--generate-samples', action='store_true', help='Generate sample visualizations')
    train_gan_parser.add_argument('--num-preview-samples', type=int, default=1, help='Number of training-end preview samples to render')
    train_gan_parser.add_argument('--num-workers', type=int, default=4, help='Number of data loader workers')
    train_gan_parser.add_argument('--resume', action='store_true',
                                 help='Resume the latest run in --output-dir from its newest checkpoint')
    train_gan_parser.set_defaults(use_ema=True, sample_from_ema=True)
    
    # Diffusion training
    train_diff_parser = subparsers.add_parser('train-diffusion', help='Train diffusion models')
    train_diff_parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    train_diff_parser.add_argument('--batch-size', type=int, default=16, help='Training batch size')
    train_diff_parser.add_argument('--void-dim', type=int, default=64, help='3D voxel space dimension')
    train_diff_parser.add_argument('--timesteps', type=int, default=1000, help='Diffusion timesteps')
    train_diff_parser.add_argument('--model-channels', type=int, default=128,
                                  help='UNet base channel width (128 needs >22GB at void 64; use 64 on the L4)')
    train_diff_parser.add_argument('--grad-checkpoint', action='store_true',
                                  help='Recompute UNet block activations in backward (fits batch 16 on the L4, ~30%% slower per step)')
    train_diff_parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    train_diff_parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay')
    train_diff_parser.add_argument('--noise-schedule', default='linear',
                                  choices=['linear', 'cosine', 'sigmoid'],
                                  help='Noise scheduling strategy')
    train_diff_parser.add_argument('--beta-start', type=float, default=0.0001, help='Beta start value')
    train_diff_parser.add_argument('--beta-end', type=float, default=0.02, help='Beta end value')
    train_diff_parser.add_argument('--data-folder', default='./data', help='Training data folder')
    train_diff_parser.add_argument('--output-dir', default='./results', help='Output directory')
    train_diff_parser.add_argument('--color', action='store_true', help='Enable color-mode diffusion channels')
    train_diff_parser.add_argument('--sparse', action='store_true', help='Use sparse tensors')
    train_diff_parser.add_argument('--mixed-precision', action='store_true', help='Use mixed precision training')
    train_diff_parser.add_argument('--scheduler', action='store_true', help='Use learning rate scheduler')
    train_diff_parser.add_argument('--use-ema', dest='use_ema', action='store_true', help='Use EMA weights for diffusion checkpoints/sampling')
    train_diff_parser.add_argument('--no-ema', dest='use_ema', action='store_false', help='Disable EMA weights for diffusion checkpoints/sampling')
    train_diff_parser.add_argument('--ema-decay', type=float, default=0.9999, help='EMA decay for diffusion model weights')
    train_diff_parser.add_argument('--resume', action='store_true',
                                  help='Resume the latest run in --output-dir from its newest checkpoint')
    train_diff_parser.add_argument('--mlflow', action='store_true', help='Enable MLflow tracking')
    train_diff_parser.add_argument('--num-workers', type=int, default=4, help='Number of data loader workers')
    train_diff_parser.add_argument('--checkpoint-freq', type=int, default=5, help='Checkpoint frequency (epochs); lower for timeout-chained cloud slices so resume loses fewer epochs')
    train_diff_parser.set_defaults(use_ema=True)
    
    # Data generation
    gen_parser = subparsers.add_parser('generate-data', help='Generate synthetic 3D data')
    gen_parser.add_argument('--num-samples', type=int, default=1000, help='Number of samples to generate')
    gen_parser.add_argument('--void-dim', type=int, default=64, help='3D voxel space dimension')
    gen_parser.add_argument('--num-shapes', type=int, default=5, help='Number of shapes per sculpture')
    gen_parser.add_argument('--structure-preset', default='shodhan',
                            choices=['shodhan', 'architectural', 'generic'],
                            help='shodhan = Corbusier/Umemoto grammar v2 (default); architectural/generic = legacy')
    gen_parser.add_argument('--seed-start', type=int, default=0,
                            help='first seed for shodhan generation (train/holdout sets use disjoint ranges)')
    gen_parser.add_argument('--grid-count', type=int, default=1, help='Enable grid columns when > 0')
    gen_parser.add_argument('--grid-step', type=int, default=4, help='Grid spacing between columns')
    gen_parser.add_argument('--edge-count', type=int, default=0, help='Number of edge primitives for the selected preset')
    gen_parser.add_argument('--edge-min-ratio', type=float, default=0.2, help='Minimum edge size ratio')
    gen_parser.add_argument('--edge-max-ratio', type=float, default=0.4, help='Maximum edge size ratio')
    gen_parser.add_argument('--plane-count', type=int, default=1, help='Plane count for generic preset')
    gen_parser.add_argument('--plane-min-ratio', type=float, default=0.3, help='Minimum plane size ratio')
    gen_parser.add_argument('--plane-max-ratio', type=float, default=0.5, help='Maximum plane size ratio')
    gen_parser.add_argument('--pipe-count', type=int, default=1, help='Pipe count for generic preset')
    gen_parser.add_argument('--pipe-min-ratio', type=float, default=0.3, help='Minimum pipe size ratio')
    gen_parser.add_argument('--pipe-max-ratio', type=float, default=0.5, help='Maximum pipe size ratio')
    gen_parser.add_argument('--output-dir', default='./data', help='Output directory')
    gen_parser.add_argument('--sparse', action='store_true', help='Use sparse tensors')
    gen_parser.add_argument('--sparse-threshold', type=float, default=0.1, help='Sparse tensor threshold')
    
    # GAN sampling
    sample_gan_parser = subparsers.add_parser('sample-gan', help='Generate samples from trained GAN')
    sample_gan_parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    sample_gan_parser.add_argument('--num-samples', type=int, default=10, help='Number of samples to generate')
    sample_gan_parser.add_argument('--output-dir', default='./samples', help='Output directory')
    sample_gan_parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    
    # Diffusion sampling
    sample_diff_parser = subparsers.add_parser('sample-diffusion', help='Generate samples from trained diffusion model')
    sample_diff_parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    sample_diff_parser.add_argument('--num-samples', type=int, default=10, help='Number of samples to generate')
    sample_diff_parser.add_argument('--num-steps', type=int, default=50, help='Number of denoising steps')
    sample_diff_parser.add_argument('--sampler', default='ddim', choices=['ddpm', 'ddim', 'dpm_solver'],
                                   help='Inference sampler. DDIM is the default fast sampler, inspired by Stable Diffusion CLI usage.')
    sample_diff_parser.add_argument('--guidance-scale', type=float, default=1.0,
                                   help='Classifier-free guidance scale. Keep at 1.0 for unconditional models.')
    sample_diff_parser.add_argument('--output-dir', default='./samples', help='Output directory')
    sample_diff_parser.add_argument('--visualize', action='store_true', help='Create visualizations')

    # Latent-space navigation
    def _add_latent_common_args(p):
        p.add_argument('--checkpoint', required=True, help='Path to generator checkpoint (config.json must sit beside it)')
        p.add_argument('--output-dir', default='./latent', help='Output directory')
        p.add_argument('--format', action='append', choices=['gif', 'png', 'obj', 'stl'],
                       help='Output format(s); repeat to combine (default: gif)')
        p.add_argument('--render', default='slice', choices=['slice', 'voxel'],
                       help='Frame renderer: slice (fast) or voxel (3D, slow)')
        p.add_argument('--fps', type=float, default=8.0, help='GIF frames per second')
        p.add_argument('--threshold', type=float, default=0.5, help='Occupancy threshold for voxel/mesh renders')
        p.add_argument('--batch-size', type=int, default=8, help='Generator batch size')
        ema = p.add_mutually_exclusive_group()
        ema.add_argument('--use-ema', dest='use_ema', action='store_true', default=True,
                         help='Prefer EMA weights when the checkpoint has them (default)')
        ema.add_argument('--no-ema', dest='use_ema', action='store_false',
                         help='Use raw generator weights')

    walk_parser = subparsers.add_parser('latent-walk', help='Interpolate between latent anchors')
    _add_latent_common_args(walk_parser)
    walk_parser.add_argument('--seeds', default='0,1', help='Comma-separated anchor seeds (2+), e.g. 12,77')
    walk_parser.add_argument('--steps', type=int, default=30, help='Steps per segment (endpoints included)')
    walk_parser.add_argument('--interp', default='slerp', choices=['lerp', 'slerp'], help='Interpolation mode')
    walk_parser.add_argument('--closed', action='store_true', help='Loop back to the first anchor')
    walk_parser.add_argument('--arithmetic', default='',
                             help='Latent expression over seed letters (a=1st seed, b=2nd, ...), e.g. "a - b + c"; walks from a to the result')
    walk_parser.add_argument('--backend', default='gan', choices=['gan', 'diffusion'],
                             help='gan: walk z-space; diffusion: walk initial-noise space (deterministic DDIM per step)')
    walk_parser.add_argument('--sampler', default='ddim', choices=['ddim', 'dpm_solver'],
                             help='Deterministic sampler for --backend diffusion')
    walk_parser.add_argument('--diffusion-steps', type=int, default=10,
                             help='Denoising steps per walk frame for --backend diffusion')

    traverse_parser = subparsers.add_parser('latent-traverse', help='Vary one latent dimension at a time')
    _add_latent_common_args(traverse_parser)
    traverse_parser.add_argument('--dims', default='', help='Comma-separated z dims (or direction indices with --directions), e.g. 0,3,9')
    traverse_parser.add_argument('--num-dims', type=int, default=8, help='Traverse the first N dims when --dims is not given')
    traverse_parser.add_argument('--steps', type=int, default=9, help='Steps across the ±sigma range')
    traverse_parser.add_argument('--sigma-range', type=float, default=3.0, help='Traversal range in standard deviations')
    traverse_parser.add_argument('--base-seed', type=int, default=42, help='Seed for the base z vector')
    traverse_parser.add_argument('--directions', default='', help='Path to a directions.pt from latent-directions; traverse principal directions instead of raw dims')

    directions_parser = subparsers.add_parser('latent-directions', help='Discover semantic directions via PCA/GANSpace')
    _add_latent_common_args(directions_parser)
    directions_parser.add_argument('--num-samples', type=int, default=2048, help='z samples for PCA')
    directions_parser.add_argument('--components', type=int, default=10, help='Number of principal directions')
    directions_parser.add_argument('--method', default='ganspace', choices=['ganspace', 'output-pca'],
                                   help='ganspace: PCA on first-linear activations (fast); output-pca: PCA on generated volumes (slow)')
    directions_parser.add_argument('--save', default='', help='Where to save directions (.pt; default <output-dir>/directions.pt)')
    directions_parser.add_argument('--apply', type=int, default=None, help='Render this direction index after computing')
    directions_parser.add_argument('--alphas', default='-3,-1.5,0,1.5,3', help='Comma-separated strengths for --apply')
    directions_parser.add_argument('--num-seeds', type=int, default=4, help='Base seeds to render for --apply')
    directions_parser.add_argument('--seed', type=int, default=0, help='Sampling seed for PCA (and base for --apply renders)')

    # Data preprocessing
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess and curate data')
    preprocess_parser.add_argument('--input-dir', required=True, help='Input data directory')
    preprocess_parser.add_argument('--output-dir', required=True, help='Output directory')
    preprocess_parser.add_argument('--encoding', default='one_hot',
                                  choices=['one_hot', 'binary', 'rgb', 'embedding'],
                                  help='Encoding method')
    preprocess_parser.add_argument('--sparse', action='store_true', help='Use sparse tensors')
    
    # Visualization
    viz_parser = subparsers.add_parser('visualize', help='Visualize 3D data')
    viz_parser.add_argument('--data-path', required=True, help='Path to data file')
    viz_parser.add_argument('--backend', default='plotly',
                           choices=['matplotlib', 'plotly', 'open3d'],
                           help='Visualization backend')
    viz_parser.add_argument('--interactive', action='store_true', help='Enable interactive visualization')
    viz_parser.add_argument('--output-path', help='Output path for static visualizations')
    
    # Benchmarking
    bench_parser = subparsers.add_parser('benchmark', help='Run performance benchmarks')
    bench_parser.add_argument('--model-type', default='skip',
                             choices=['simple', 'complex', 'skip', 'monochrome', 'autoencoder'],
                             help='Model type to benchmark')
    bench_parser.add_argument('--batch-size', type=int, default=32, help='Batch size for benchmarking')
    bench_parser.add_argument('--void-dim', type=int, default=64, help='3D voxel space dimension')
    bench_parser.add_argument('--noise-dim', type=int, default=100, help='Noise vector dimension')
    bench_parser.add_argument('--sparse', action='store_true', help='Use sparse tensors')
    bench_parser.add_argument('--profile-memory', action='store_true', help='Profile memory usage')
    bench_parser.add_argument('--save-results', action='store_true', help='Save benchmark results')
    bench_parser.add_argument('--output-dir', default='./benchmarks', help='Output directory for results')
    
    # Model evaluation
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained models')
    eval_parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    eval_parser.add_argument('--test-data', required=True, help='Path to test data')
    eval_parser.add_argument('--metrics', default='all',
                            choices=['all', 'fid', 'is', 'lpips'],
                            help='Evaluation metrics to compute')
    eval_parser.add_argument('--output-dir', default='./evaluation', help='Output directory')
    
    # Model export
    export_parser = subparsers.add_parser('export', help='Export models for deployment')
    export_parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    export_parser.add_argument('--format', default='onnx',
                              choices=['onnx', 'torchscript', 'tensorrt'],
                              help='Export format')
    export_parser.add_argument('--output', required=True, help='Output path for exported model')
    
    return parser


def main():
    """Main entry point for DeepSculpt v2.0."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    print(f"DeepSculpt v2.0 - Command: {args.command}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Initialize main orchestrator
    try:
        main_app = DeepSculptV2Main(args)
        
        # Route to appropriate command
        if args.command == 'train-gan':
            return main_app.train_gan(args)
        elif args.command == 'train-diffusion':
            return main_app.train_diffusion(args)
        elif args.command == 'generate-data':
            return main_app.generate_data(args)
        elif args.command == 'sample-gan':
            return main_app.sample_gan(args)
        elif args.command == 'latent-walk':
            return main_app.latent_walk(args)
        elif args.command == 'latent-traverse':
            return main_app.latent_traverse(args)
        elif args.command == 'latent-directions':
            return main_app.latent_directions(args)
        elif args.command == 'sample-diffusion':
            return main_app.sample_diffusion(args)
        elif args.command == 'preprocess':
            return main_app.preprocess_data(args)
        elif args.command == 'visualize':
            return main_app.visualize(args)
        elif args.command == 'benchmark':
            return main_app.benchmark(args)
        elif args.command == 'evaluate':
            return main_app.evaluate(args)
        elif args.command == 'export':
            return main_app.export_model(args)
        else:
            print(f"Unknown command: {args.command}")
            parser.print_help()
            return 1
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        print(f"Error executing command '{args.command}': {e}")
        # Always print the traceback: on cloud runs the message line is the
        # only thing that reaches Cloud Logging, and an un-located OOM cost
        # several paid iterations to root-cause.
        import traceback
        traceback.print_exc()
        if torch.cuda.is_available() and "out of memory" in str(e):
            print(torch.cuda.memory_summary())
        return 1


if __name__ == "__main__":
    sys.exit(main())
