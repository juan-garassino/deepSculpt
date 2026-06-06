"""
Comprehensive tests for PyTorch training infrastructure.
Tests training convergence, stability, and all major functionality.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

from deepsculpt.core.training.gan_trainer import (
    TrainingConfig, BaseTrainer, GANTrainer, DiffusionTrainer,
    NoiseScheduler, EarlyStopping, ModelCheckpointManager,
    TrainingMonitor, HyperparameterTuner, DistributedTrainingManager,
    TrainingOrchestrator, create_optimizer, create_scheduler,
    create_training_config, setup_training_environment
)


class SimpleTestModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self, input_dim: int = 100, output_dim: int = 64):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.fc(x)


class SimpleGenerator(nn.Module):
    """Simple generator for GAN testing."""
    
    def __init__(self, noise_dim: int = 100, output_dim: int = 64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.fc(x).view(-1, 4, 4, 4, 1)  # Reshape to 3D volume


class SimpleDiscriminator(nn.Module):
    """Simple discriminator for GAN testing."""
    
    def __init__(self, input_dim: int = 64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Flatten 3D volume
        x = x.view(x.size(0), -1)
        return self.fc(x)


class SimpleDiffusionModel(nn.Module):
    """Simple diffusion model for testing."""
    
    def __init__(self, input_dim: int = 64):
        super().__init__()
        self.time_embed = nn.Embedding(1000, 128)
        self.fc = nn.Sequential(
            nn.Linear(input_dim + 128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )
    
    def forward(self, x, timesteps, conditioning=None):
        # Flatten input
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        
        # Time embedding
        t_emb = self.time_embed(timesteps)
        
        # Concatenate
        combined = torch.cat([x_flat, t_emb], dim=1)
        
        # Process
        output = self.fc(combined)
        
        # Reshape back
        return output.view_as(x)


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def training_config(temp_dir):
    """Create test training configuration."""
    return TrainingConfig(
        batch_size=4,
        learning_rate=0.001,
        epochs=5,
        mixed_precision=False,  # Disable for testing
        checkpoint_freq=2,
        log_freq=1,
        checkpoint_dir=str(Path(temp_dir) / "checkpoints"),
        log_dir=str(Path(temp_dir) / "logs"),
        snapshot_dir=str(Path(temp_dir) / "snapshots"),
        use_tensorboard=False,  # Disable for testing
        use_wandb=False,
        use_mlflow=False
    )


@pytest.fixture
def sample_data():
    """Create sample training data."""
    batch_size = 4
    data_shape = (4, 4, 4, 1)  # Small 3D volumes
    
    # Create synthetic data
    data = torch.randn(batch_size, *data_shape)
    return data


class TestTrainingConfig:
    """Test training configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = TrainingConfig()
        
        assert config.batch_size == 32
        assert config.learning_rate == 0.0002
        assert config.epochs == 100
        assert config.mixed_precision == True
        assert config.distributed == False
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = TrainingConfig(
            batch_size=16,
            learning_rate=0.001,
            epochs=50
        )
        
        assert config.batch_size == 16
        assert config.learning_rate == 0.001
        assert config.epochs == 50
    
    def test_create_training_config(self):
        """Test factory function for creating config."""
        config = create_training_config(
            batch_size=8,
            learning_rate=0.0001
        )
        
        assert isinstance(config, TrainingConfig)
        assert config.batch_size == 8
        assert config.learning_rate == 0.0001


class TestNoiseScheduler:
    """Test noise scheduler for diffusion models."""
    
    def test_linear_schedule(self):
        """Test linear noise schedule."""
        scheduler = NoiseScheduler(
            schedule_type="linear",
            timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            device="cpu"
        )
        
        assert scheduler.betas.shape == (100,)
        assert scheduler.alphas.shape == (100,)
        assert scheduler.alphas_cumprod.shape == (100,)
        
        # Check that betas increase linearly
        assert scheduler.betas[0] < scheduler.betas[-1]
        assert torch.allclose(scheduler.betas[0], torch.tensor(0.0001), atol=1e-6)
        assert torch.allclose(scheduler.betas[-1], torch.tensor(0.02), atol=1e-6)
    
    def test_cosine_schedule(self):
        """Test cosine noise schedule."""
        scheduler = NoiseScheduler(
            schedule_type="cosine",
            timesteps=100,
            device="cpu"
        )
        
        assert scheduler.betas.shape == (100,)
        assert scheduler.alphas.shape == (100,)
        assert scheduler.alphas_cumprod.shape == (100,)
        
        # Check that all values are in valid range
        assert torch.all(scheduler.betas >= 0)
        assert torch.all(scheduler.betas <= 1)
        assert torch.all(scheduler.alphas >= 0)
        assert torch.all(scheduler.alphas <= 1)
    
    def test_add_noise(self):
        """Test noise addition."""
        scheduler = NoiseScheduler(timesteps=100, device="cpu")
        
        # Create test data
        original_samples = torch.randn(2, 4, 4, 4, 1)
        noise = torch.randn_like(original_samples)
        timesteps = torch.randint(0, 100, (2,))
        
        # Add noise
        noisy_samples = scheduler.add_noise(original_samples, noise, timesteps)
        
        assert noisy_samples.shape == original_samples.shape
        assert not torch.allclose(noisy_samples, original_samples)
    
    def test_get_velocity(self):
        """Test velocity calculation for v-parameterization."""
        scheduler = NoiseScheduler(timesteps=100, device="cpu")
        
        sample = torch.randn(2, 4, 4, 4, 1)
        noise = torch.randn_like(sample)
        timesteps = torch.randint(0, 100, (2,))
        
        velocity = scheduler.get_velocity(sample, noise, timesteps)
        
        assert velocity.shape == sample.shape


class TestGANTrainer:
    """Test GAN trainer functionality."""
    
    def test_gan_trainer_initialization(self, training_config):
        """Test GAN trainer initialization."""
        generator = SimpleGenerator()
        discriminator = SimpleDiscriminator()
        gen_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
        disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)
        
        trainer = GANTrainer(
            generator=generator,
            discriminator=discriminator,
            gen_optimizer=gen_optimizer,
            disc_optimizer=disc_optimizer,
            config=training_config,
            device="cpu"
        )
        
        assert trainer.generator is generator
        assert trainer.discriminator is discriminator
        assert trainer.noise_dim == 100
        assert trainer.fixed_noise.shape == (16, 100)
    
    def test_adversarial_loss(self, training_config):
        """Test adversarial loss calculation."""
        generator = SimpleGenerator()
        discriminator = SimpleDiscriminator()
        gen_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
        disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)
        
        trainer = GANTrainer(
            generator=generator,
            discriminator=discriminator,
            gen_optimizer=gen_optimizer,
            disc_optimizer=disc_optimizer,
            config=training_config,
            device="cpu"
        )
        
        # Test real target
        output = torch.randn(4, 1)
        real_loss = trainer.adversarial_loss(output, True)
        assert real_loss.item() >= 0
        
        # Test fake target
        fake_loss = trainer.adversarial_loss(output, False)
        assert fake_loss.item() >= 0
    
    def test_train_step(self, training_config, sample_data):
        """Test single training step."""
        generator = SimpleGenerator()
        discriminator = SimpleDiscriminator()
        gen_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
        disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)
        
        trainer = GANTrainer(
            generator=generator,
            discriminator=discriminator,
            gen_optimizer=gen_optimizer,
            disc_optimizer=disc_optimizer,
            config=training_config,
            device="cpu"
        )
        
        # Perform training step
        metrics = trainer.train_step(sample_data)
        
        # Check that metrics are returned
        assert 'gen_loss' in metrics
        assert 'disc_loss' in metrics
        assert 'disc_real_acc' in metrics
        assert 'disc_fake_acc' in metrics
        
        # Check that losses are reasonable
        assert metrics['gen_loss'] >= 0
        assert metrics['disc_loss'] >= 0
        assert 0 <= metrics['disc_real_acc'] <= 1
        assert 0 <= metrics['disc_fake_acc'] <= 1
    
    def test_generate_samples(self, training_config):
        """Test sample generation."""
        generator = SimpleGenerator()
        discriminator = SimpleDiscriminator()
        gen_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
        disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)
        
        trainer = GANTrainer(
            generator=generator,
            discriminator=discriminator,
            gen_optimizer=gen_optimizer,
            disc_optimizer=disc_optimizer,
            config=training_config,
            device="cpu"
        )
        
        # Generate samples
        samples = trainer.generate_samples(num_samples=8)
        
        assert samples.shape == (8, 4, 4, 4, 1)
        assert samples.dtype == torch.float32


class TestDiffusionTrainer:
    """Test diffusion trainer functionality."""
    
    def test_diffusion_trainer_initialization(self, training_config):
        """Test diffusion trainer initialization."""
        model = SimpleDiffusionModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        noise_scheduler = NoiseScheduler(timesteps=100, device="cpu")
        
        trainer = DiffusionTrainer(
            model=model,
            optimizer=optimizer,
            config=training_config,
            noise_scheduler=noise_scheduler,
            device="cpu"
        )
        
        assert trainer.model is model
        assert trainer.noise_scheduler is noise_scheduler
        assert trainer.prediction_type == "epsilon"
    
    def test_compute_loss(self, training_config):
        """Test loss computation."""
        model = SimpleDiffusionModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        noise_scheduler = NoiseScheduler(timesteps=100, device="cpu")
        
        trainer = DiffusionTrainer(
            model=model,
            optimizer=optimizer,
            config=training_config,
            noise_scheduler=noise_scheduler,
            device="cpu"
        )
        
        # Create test data
        model_output = torch.randn(2, 4, 4, 4, 1)
        target = torch.randn(2, 4, 4, 4, 1)
        timesteps = torch.randint(0, 100, (2,))
        sample = torch.randn(2, 4, 4, 4, 1)
        
        losses = trainer.compute_loss(model_output, target, timesteps, sample)
        
        assert 'diffusion_loss' in losses
        assert 'mse_loss' in losses
        assert 'l1_loss' in losses
        
        assert losses['diffusion_loss'].item() >= 0
        assert losses['mse_loss'].item() >= 0
        assert losses['l1_loss'].item() >= 0
    
    def test_train_step(self, training_config, sample_data):
        """Test single training step."""
        model = SimpleDiffusionModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        noise_scheduler = NoiseScheduler(timesteps=100, device="cpu")
        
        trainer = DiffusionTrainer(
            model=model,
            optimizer=optimizer,
            config=training_config,
            noise_scheduler=noise_scheduler,
            device="cpu"
        )
        
        # Perform training step
        metrics = trainer.train_step(sample_data)
        
        # Check that metrics are returned
        assert 'diffusion_loss' in metrics
        assert 'mse_loss' in metrics
        assert 'l1_loss' in metrics
        
        # Check that losses are reasonable
        assert metrics['diffusion_loss'] >= 0
        assert metrics['mse_loss'] >= 0
        assert metrics['l1_loss'] >= 0
    
    def test_sample_generation(self, training_config):
        """Test sample generation."""
        model = SimpleDiffusionModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        noise_scheduler = NoiseScheduler(timesteps=100, device="cpu")
        
        trainer = DiffusionTrainer(
            model=model,
            optimizer=optimizer,
            config=training_config,
            noise_scheduler=noise_scheduler,
            device="cpu"
        )
        
        # Generate samples
        shape = (2, 4, 4, 4, 1)
        samples = trainer.sample(shape, num_inference_steps=10)
        
        assert samples.shape == shape
        assert samples.dtype == torch.float32
    
    def test_ddim_sampling(self, training_config):
        """Test DDIM sampling."""
        model = SimpleDiffusionModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        noise_scheduler = NoiseScheduler(timesteps=100, device="cpu")
        
        trainer = DiffusionTrainer(
            model=model,
            optimizer=optimizer,
            config=training_config,
            noise_scheduler=noise_scheduler,
            device="cpu"
        )
        
        # Generate samples with DDIM
        shape = (2, 4, 4, 4, 1)
        samples = trainer.sample_ddim(shape, num_inference_steps=10)
        
        assert samples.shape == shape
        assert samples.dtype == torch.float32


class TestTrainingInfrastructure:
    """Test training infrastructure components."""
    
    def test_early_stopping(self):
        """Test early stopping mechanism."""
        early_stopping = EarlyStopping(patience=3, min_delta=0.01, mode='min')
        
        # Simulate improving scores
        assert not early_stopping(1.0)
        assert not early_stopping(0.9)
        assert not early_stopping(0.8)
        
        # Simulate stagnating scores
        assert not early_stopping(0.81)  # Within min_delta
        assert not early_stopping(0.82)
        assert not early_stopping(0.83)
        assert early_stopping(0.84)  # Should trigger early stopping
    
    def test_checkpoint_manager(self, temp_dir):
        """Test checkpoint management."""
        checkpoint_dir = str(Path(temp_dir) / "checkpoints")
        manager = ModelCheckpointManager(
            checkpoint_dir=checkpoint_dir,
            max_checkpoints=3,
            save_best=True,
            monitor='val_loss',
            mode='min'
        )
        
        # Create mock trainer
        model = SimpleTestModel()
        optimizer = torch.optim.Adam(model.parameters())
        config = TrainingConfig()
        trainer = BaseTrainer(model, optimizer, config, device="cpu")
        
        # Save some checkpoints
        manager.save_checkpoint(trainer, 1, {'val_loss': 1.0})
        manager.save_checkpoint(trainer, 2, {'val_loss': 0.8})
        manager.save_checkpoint(trainer, 3, {'val_loss': 1.2})
        
        # Check that best checkpoint is tracked
        best_checkpoint = manager.get_best_checkpoint()
        assert best_checkpoint is not None
        assert "epoch_0002" in best_checkpoint  # Epoch 2 had lowest loss
    
    def test_training_monitor(self):
        """Test training monitoring."""
        monitor = TrainingMonitor(window_size=5)
        
        monitor.start_training()
        
        # Log some epochs
        for epoch in range(10):
            metrics = {
                'train_loss': 1.0 - epoch * 0.1,
                'val_loss': 1.2 - epoch * 0.08
            }
            monitor.log_epoch(epoch, metrics, 1.0)
        
        # Get training stats
        stats = monitor.get_training_stats()
        
        assert 'total_training_time' in stats
        assert 'avg_epoch_time' in stats
        assert 'train_loss_current' in stats
        assert 'val_loss_trend' in stats
        
        # Check trend calculation
        assert stats['train_loss_trend'] == 'decreasing'
        assert stats['val_loss_trend'] == 'decreasing'
    
    def test_hyperparameter_tuner(self):
        """Test hyperparameter tuning."""
        param_ranges = {
            'learning_rate': (1e-5, 1e-2),
            'batch_size': (16, 128),
            'beta1': (0.5, 0.9)
        }
        
        tuner = HyperparameterTuner(param_ranges)
        
        # Suggest parameters
        params = tuner.suggest_parameters(method="random")
        
        assert 'learning_rate' in params
        assert 'batch_size' in params
        assert 'beta1' in params
        
        assert 1e-5 <= params['learning_rate'] <= 1e-2
        assert 16 <= params['batch_size'] <= 128
        assert 0.5 <= params['beta1'] <= 0.9
        
        # Record trial
        tuner.record_trial(params, 0.5)
        
        # Get best parameters
        best_params = tuner.get_best_parameters()
        assert best_params == params


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_create_optimizer(self):
        """Test optimizer creation."""
        model = SimpleTestModel()
        
        # Test Adam optimizer
        optimizer = create_optimizer(model, "adam", 0.001)
        assert isinstance(optimizer, torch.optim.Adam)
        assert optimizer.param_groups[0]['lr'] == 0.001
        
        # Test AdamW optimizer
        optimizer = create_optimizer(model, "adamw", 0.001, weight_decay=0.01)
        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.param_groups[0]['weight_decay'] == 0.01
        
        # Test SGD optimizer
        optimizer = create_optimizer(model, "sgd", 0.01, momentum=0.9)
        assert isinstance(optimizer, torch.optim.SGD)
        assert optimizer.param_groups[0]['momentum'] == 0.9
    
    def test_create_scheduler(self):
        """Test scheduler creation."""
        model = SimpleTestModel()
        optimizer = torch.optim.Adam(model.parameters())
        
        # Test cosine scheduler
        scheduler = create_scheduler(optimizer, "cosine", T_max=100)
        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
        
        # Test step scheduler
        scheduler = create_scheduler(optimizer, "step", step_size=30)
        assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)
        
        # Test exponential scheduler
        scheduler = create_scheduler(optimizer, "exponential", gamma=0.95)
        assert isinstance(scheduler, torch.optim.lr_scheduler.ExponentialLR)
    
    def test_setup_training_environment(self, temp_dir):
        """Test training environment setup."""
        config = TrainingConfig(
            checkpoint_dir=str(Path(temp_dir) / "checkpoints"),
            log_dir=str(Path(temp_dir) / "logs"),
            snapshot_dir=str(Path(temp_dir) / "snapshots")
        )
        
        env = setup_training_environment(config)
        
        assert 'device' in env
        assert 'checkpoint_dir' in env
        assert 'log_dir' in env
        assert 'snapshot_dir' in env
        
        # Check that directories were created
        assert Path(config.checkpoint_dir).exists()
        assert Path(config.log_dir).exists()
        assert Path(config.snapshot_dir).exists()


class TestTrainingOrchestrator:
    """Test training orchestrator."""
    
    def test_orchestrator_initialization(self, training_config):
        """Test orchestrator initialization."""
        orchestrator = TrainingOrchestrator(training_config)
        
        assert orchestrator.config is training_config
        assert orchestrator.checkpoint_manager is not None
        assert orchestrator.training_monitor is not None
        assert orchestrator.early_stopping is not None
    
    @pytest.mark.slow
    def test_gan_training_orchestration(self, training_config, sample_data):
        """Test GAN training orchestration."""
        # Create simple dataset
        dataset = torch.utils.data.TensorDataset(sample_data)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
        
        # Create models
        generator = SimpleGenerator()
        discriminator = SimpleDiscriminator()
        
        # Create orchestrator
        orchestrator = TrainingOrchestrator(training_config)
        
        # Run training (short)
        training_config.epochs = 2
        results = orchestrator.train_gan(
            generator=generator,
            discriminator=discriminator,
            train_dataloader=dataloader
        )
        
        assert 'best_epoch' in results
        assert 'final_metrics' in results
        assert 'training_stats' in results
    
    @pytest.mark.slow
    def test_diffusion_training_orchestration(self, training_config, sample_data):
        """Test diffusion training orchestration."""
        # Create simple dataset
        dataset = torch.utils.data.TensorDataset(sample_data)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
        
        # Create model
        model = SimpleDiffusionModel()
        
        # Create orchestrator
        orchestrator = TrainingOrchestrator(training_config)
        
        # Run training (short)
        training_config.epochs = 2
        results = orchestrator.train_diffusion(
            model=model,
            train_dataloader=dataloader
        )
        
        assert 'best_epoch' in results
        assert 'final_metrics' in results
        assert 'training_stats' in results


class TestIntegration:
    """Integration tests for complete training workflows."""
    
    @pytest.mark.slow
    def test_complete_gan_workflow(self, temp_dir):
        """Test complete GAN training workflow."""
        # Setup
        config = TrainingConfig(
            batch_size=4,
            learning_rate=0.001,
            epochs=3,
            mixed_precision=False,
            checkpoint_freq=1,
            checkpoint_dir=str(Path(temp_dir) / "checkpoints"),
            log_dir=str(Path(temp_dir) / "logs"),
            use_tensorboard=False,
            use_wandb=False,
            use_mlflow=False
        )
        
        # Create synthetic data
        data = torch.randn(16, 4, 4, 4, 1)
        dataset = torch.utils.data.TensorDataset(data)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)
        
        # Create models
        generator = SimpleGenerator()
        discriminator = SimpleDiscriminator()
        
        # Create optimizers
        gen_optimizer = create_optimizer(generator, "adam", config.learning_rate)
        disc_optimizer = create_optimizer(discriminator, "adam", config.learning_rate)
        
        # Create trainer
        trainer = GANTrainer(
            generator=generator,
            discriminator=discriminator,
            gen_optimizer=gen_optimizer,
            disc_optimizer=disc_optimizer,
            config=config,
            device="cpu"
        )
        
        # Train for a few epochs
        for epoch in range(config.epochs):
            train_metrics = trainer.train_epoch(train_loader)
            val_metrics = trainer.validate(val_loader)
            
            # Check that training produces reasonable metrics
            assert 'gen_loss' in train_metrics
            assert 'disc_loss' in train_metrics
            assert train_metrics['gen_loss'] >= 0
            assert train_metrics['disc_loss'] >= 0
            
            # Save checkpoint
            checkpoint_path = Path(temp_dir) / f"checkpoint_epoch_{epoch}.pth"
            trainer.save_checkpoint(str(checkpoint_path), epoch, train_metrics)
            
            # Verify checkpoint was saved
            assert checkpoint_path.exists()
        
        # Test sample generation
        samples = trainer.generate_samples(num_samples=4)
        assert samples.shape == (4, 4, 4, 4, 1)
        
        # Test checkpoint loading
        new_trainer = GANTrainer(
            generator=SimpleGenerator(),
            discriminator=SimpleDiscriminator(),
            gen_optimizer=create_optimizer(SimpleGenerator(), "adam", config.learning_rate),
            disc_optimizer=create_optimizer(SimpleDiscriminator(), "adam", config.learning_rate),
            config=config,
            device="cpu"
        )
        
        checkpoint_data = new_trainer.load_checkpoint(str(checkpoint_path))
        assert checkpoint_data['epoch'] == epoch
    
    @pytest.mark.slow
    def test_complete_diffusion_workflow(self, temp_dir):
        """Test complete diffusion training workflow."""
        # Setup
        config = TrainingConfig(
            batch_size=4,
            learning_rate=0.001,
            epochs=3,
            mixed_precision=False,
            checkpoint_freq=1,
            checkpoint_dir=str(Path(temp_dir) / "checkpoints"),
            log_dir=str(Path(temp_dir) / "logs"),
            use_tensorboard=False,
            use_wandb=False,
            use_mlflow=False
        )
        
        # Create synthetic data
        data = torch.randn(16, 4, 4, 4, 1)
        dataset = torch.utils.data.TensorDataset(data)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)
        
        # Create model and components
        model = SimpleDiffusionModel()
        optimizer = create_optimizer(model, "adamw", config.learning_rate)
        noise_scheduler = NoiseScheduler(timesteps=100, device="cpu")
        
        # Create trainer
        trainer = DiffusionTrainer(
            model=model,
            optimizer=optimizer,
            config=config,
            noise_scheduler=noise_scheduler,
            device="cpu"
        )
        
        # Train for a few epochs
        for epoch in range(config.epochs):
            train_metrics = trainer.train_epoch(train_loader)
            val_metrics = trainer.validate(val_loader)
            
            # Check that training produces reasonable metrics
            assert 'diffusion_loss' in train_metrics
            assert 'mse_loss' in train_metrics
            assert train_metrics['diffusion_loss'] >= 0
            assert train_metrics['mse_loss'] >= 0
            
            # Save checkpoint
            checkpoint_path = Path(temp_dir) / f"diffusion_checkpoint_epoch_{epoch}.pth"
            trainer.save_checkpoint(str(checkpoint_path), epoch, train_metrics)
            
            # Verify checkpoint was saved
            assert checkpoint_path.exists()
        
        # Test sample generation
        shape = (2, 4, 4, 4, 1)
        samples = trainer.sample(shape, num_inference_steps=5)
        assert samples.shape == shape
        
        # Test DDIM sampling
        ddim_samples = trainer.sample_ddim(shape, num_inference_steps=5)
        assert ddim_samples.shape == shape
        
        # Test checkpoint loading
        new_trainer = DiffusionTrainer(
            model=SimpleDiffusionModel(),
            optimizer=create_optimizer(SimpleDiffusionModel(), "adamw", config.learning_rate),
            config=config,
            noise_scheduler=NoiseScheduler(timesteps=100, device="cpu"),
            device="cpu"
        )
        
        checkpoint_data = new_trainer.load_checkpoint(str(checkpoint_path))
        assert checkpoint_data['epoch'] == epoch


if __name__ == "__main__":
    pytest.main([__file__, "-v"])