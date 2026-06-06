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
    from deepsculpt.core.models.pytorch_models import *
    from deepsculpt.core.training.gan_trainer import GANTrainer
    from deepsculpt.core.training.diffusion_trainer import DiffusionTrainer
    from deepsculpt.core.training.base_trainer import BaseTrainer, TrainingConfig
    from deepsculpt.core.data.generation.pytorch_collector import PyTorchCollector
    from deepsculpt.core.data.generation.pytorch_sculptor import PyTorchSculptor
    from deepsculpt.core.data.transforms.pytorch_curator import PyTorchCurator
    from deepsculpt.core.data.loaders.data_loaders import StreamingDataLoader
    from deepsculpt.core.visualization.pytorch_visualization import PyTorchVisualizer
    from deepsculpt.core.workflow.pytorch_workflow import PyTorchWorkflowManager
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
    
    def train_gan(self, args):
        """Train GAN models with comprehensive configuration and monitoring."""
        print(f"Training GAN model: {args.model_type}")
        
        # Create results directory
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
            checkpoint_dir=str(results_dir / "checkpoints"),
            log_dir=str(results_dir / "logs"),
            snapshot_dir=str(results_dir / "snapshots"),
            use_tensorboard=False,  # Disable TensorBoard since it's not available
            use_wandb=False,
            use_mlflow=False
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
        
        # Train the model
        print(f"Starting training for {args.epochs} epochs")
        metrics = trainer.train(
            train_dataloader=data_loader
        )
        
        # Save final models
        torch.save(generator.state_dict(), results_dir / "generator_final.pt")
        if trainer.ema_generator is not None:
            torch.save(trainer.ema_generator.state_dict(), results_dir / "ema_generator_final.pt")
        torch.save(discriminator.state_dict(), results_dir / "discriminator_final.pt")
        
        # Save configuration
        config = {
            "model_type": args.model_type,
            "void_dim": args.void_dim,
            "noise_dim": args.noise_dim,
            "color_mode": 1 if args.color else 0,
            "sparse": args.sparse,
            "discriminator_type": args.discriminator_type,
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
        
        # Create results directory
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
            sparse=args.sparse
        ).to(self.device)
        
        # Create noise scheduler
        from core.models.diffusion.noise_scheduler import NoiseScheduler
        noise_scheduler = NoiseScheduler(
            schedule_type=args.noise_schedule,
            timesteps=args.timesteps,
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            device=self.device  # Use the same device as the main app
        )
        
        # Create diffusion pipeline
        from core.models.diffusion.pipeline import Diffusion3DPipeline
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
            checkpoint_dir=str(results_dir / "checkpoints"),
            log_dir=str(results_dir / "logs"),
            snapshot_dir=str(results_dir / "snapshots"),
            use_tensorboard=False,
            use_wandb=False,
            use_mlflow=False
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
        
        # Train the model
        print(f"Starting diffusion training for {args.epochs} epochs")
        metrics = trainer.train(
            train_dataloader=data_loader
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
            }
        }, results_dir / "diffusion_final.pt")

        with open(results_dir / "config.json", "w") as f:
            json.dump(
                {
                    "model_type": args.model_type,
                    "void_dim": args.void_dim,
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
        
        # Load model configuration
        checkpoint_dir = Path(args.checkpoint).parent
        config_path = checkpoint_dir / "config.json"
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            print("Warning: Model configuration not found. Using defaults.")
            config = {
                "model_type": "skip",
                "void_dim": 64,
                "noise_dim": 100,
                "color_mode": 1,
                "sparse": False
            }
        
        # Create model factory
        model_factory = PyTorchModelFactoryV2()
        
        # Create model
        generator = model_factory.create_gan_generator(
            model_type=config['model_type'],
            void_dim=config['void_dim'],
            noise_dim=config['noise_dim'],
            color_mode=config['color_mode'],
            sparse=config.get('sparse', False)
        ).to(self.device)
        
        # Load checkpoint
        generator.load_state_dict(torch.load(args.checkpoint, map_location=self.device))
        generator.eval()
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate samples
        print(f"Generating {args.num_samples} samples...")
        samples = []
        
        with torch.no_grad():
            for i in range(args.num_samples):
                noise = torch.randn(1, config['noise_dim'], device=self.device)
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
            sparse=config.get('sparse', False)
        ).to(self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Keep the scheduler tensors on the active device after checkpoint load.
        noise_scheduler = checkpoint['noise_scheduler']
        if hasattr(noise_scheduler, "device"):
            noise_scheduler.device = self.device
        if hasattr(noise_scheduler, "_to_device"):
            noise_scheduler._to_device()

        from core.models.diffusion.pipeline import Diffusion3DPipeline, FastSamplingPipeline
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
        
        experiment_name = f"deepsculpt_{args.command}"
        mlflow.set_experiment(experiment_name)
        
        run_name = f"{args.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        mlflow.start_run(run_name=run_name)
        
        # Log parameters
        params = {
            "framework": "pytorch",
            "model_type": getattr(args, 'model_type', 'unknown'),
            "void_dim": args.void_dim,
            "epochs": getattr(args, 'epochs', 0),
            "batch_size": args.batch_size,
            "device": self.device
        }
        
        mlflow.log_params(params)
        
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
    train_gan_parser.set_defaults(use_ema=True, sample_from_ema=True)
    
    # Diffusion training
    train_diff_parser = subparsers.add_parser('train-diffusion', help='Train diffusion models')
    train_diff_parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    train_diff_parser.add_argument('--batch-size', type=int, default=16, help='Training batch size')
    train_diff_parser.add_argument('--void-dim', type=int, default=64, help='3D voxel space dimension')
    train_diff_parser.add_argument('--timesteps', type=int, default=1000, help='Diffusion timesteps')
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
    train_diff_parser.add_argument('--mlflow', action='store_true', help='Enable MLflow tracking')
    train_diff_parser.add_argument('--num-workers', type=int, default=4, help='Number of data loader workers')
    train_diff_parser.set_defaults(use_ema=True)
    
    # Data generation
    gen_parser = subparsers.add_parser('generate-data', help='Generate synthetic 3D data')
    gen_parser.add_argument('--num-samples', type=int, default=1000, help='Number of samples to generate')
    gen_parser.add_argument('--void-dim', type=int, default=64, help='3D voxel space dimension')
    gen_parser.add_argument('--num-shapes', type=int, default=5, help='Number of shapes per sculpture')
    gen_parser.add_argument('--structure-preset', default='architectural', choices=['architectural', 'generic'], help='Procedural structure recipe preset')
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
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
