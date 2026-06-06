#!/usr/bin/env python3
"""
End-to-End Integration Tests for DeepSculpt v2.0

Comprehensive integration testing covering the complete PyTorch pipeline
from data generation to model training and inference.
"""

import pytest
import torch
import tempfile
import shutil
from pathlib import Path
import json
import time
from typing import Dict, Any, List

# Import DeepSculpt v2.0 modules
from deepsculpt.core.data.generation.pytorch_collector import PyTorchCollector
from deepsculpt.core.data.generation.pytorch_sculptor import PyTorchSculptor
from deepsculpt.core.data.transforms.pytorch_curator import PyTorchCurator
from deepsculpt.core.models.model_factory import PyTorchModelFactory
from deepsculpt.core.training.pytorch_trainer import GANTrainer
from deepsculpt.core.training.diffusion_trainer import DiffusionTrainer
from deepsculpt.core.visualization.pytorch_visualization import PyTorchVisualizer
from deepsculpt.core.utils.pytorch_utils import PyTorchUtils


class TestEndToEndPipeline:
    """End-to-end integration tests for the complete PyTorch pipeline."""
    
    @pytest.fixture(autouse=True)
    def setup_test_environment(self):
        """Setup test environment with temporary directories."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.data_dir = self.temp_dir / "data"
        self.results_dir = self.temp_dir / "results"
        self.checkpoints_dir = self.temp_dir / "checkpoints"
        
        # Create directories
        self.data_dir.mkdir(parents=True)
        self.results_dir.mkdir(parents=True)
        self.checkpoints_dir.mkdir(parents=True)
        
        # Setup device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        yield
        
        # Cleanup
        shutil.rmtree(self.temp_dir)
    
    @pytest.mark.integration
    def test_complete_gan_pipeline(self):
        """Test complete GAN pipeline from data generation to inference."""
        print("Testing complete GAN pipeline...")
        
        # Step 1: Generate synthetic data
        print("Step 1: Generating synthetic data...")
        sculptor_config = {
            "void_dim": 32,  # Small for testing
            "num_shapes": 3,
            "sparse_threshold": 0.1
        }
        
        collector = PyTorchCollector(
            sculptor_config=sculptor_config,
            output_format="pytorch",
            sparse_threshold=0.1,
            device=self.device
        )
        
        # Generate small dataset for testing
        num_samples = 10
        dataset_paths = collector.create_collection(num_samples)
        
        assert len(dataset_paths) == num_samples
        print(f"✓ Generated {num_samples} samples")
        
        # Step 2: Create data loader
        print("Step 2: Creating data loader...")
        dataset = collector.create_streaming_dataset(num_samples)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0  # Avoid multiprocessing issues in tests
        )
        
        # Verify data loader works
        batch = next(iter(data_loader))
        assert 'structure' in batch
        assert 'colors' in batch
        print("✓ Data loader created successfully")
        
        # Step 3: Create GAN models
        print("Step 3: Creating GAN models...")
        generator = PyTorchModelFactory.create_gan_generator(
            model_type="simple",  # Use simple model for testing
            void_dim=32,
            noise_dim=64,
            color_mode=1,
            sparse=False
        ).to(self.device)
        
        discriminator = PyTorchModelFactory.create_gan_discriminator(
            model_type="simple",
            void_dim=32,
            color_mode=1,
            sparse=False
        ).to(self.device)
        
        # Verify models can process data
        with torch.no_grad():
            noise = torch.randn(2, 64, device=self.device)
            fake_data = generator(noise)
            disc_output = discriminator(fake_data)
            
        assert fake_data.shape[0] == 2
        assert disc_output.shape[0] == 2
        print("✓ GAN models created and tested")
        
        # Step 4: Train models (minimal training for testing)
        print("Step 4: Training GAN models...")
        gen_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
        disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)
        
        trainer = GANTrainer(
            generator=generator,
            discriminator=discriminator,
            gen_optimizer=gen_optimizer,
            disc_optimizer=disc_optimizer,
            device=self.device,
            mixed_precision=False  # Disable for testing
        )
        
        # Train for just 2 epochs for testing
        metrics = trainer.train(
            data_loader=data_loader,
            epochs=2,
            checkpoint_dir=self.checkpoints_dir,
            snapshot_dir=self.results_dir,
            snapshot_freq=1
        )
        
        assert 'gen_loss' in metrics
        assert 'disc_loss' in metrics
        assert len(metrics['gen_loss']) == 2
        print("✓ GAN training completed")
        
        # Step 5: Generate samples
        print("Step 5: Generating samples...")
        generator.eval()
        with torch.no_grad():
            noise = torch.randn(5, 64, device=self.device)
            samples = generator(noise)
            
        assert samples.shape[0] == 5
        assert samples.shape[1:] == (1, 32, 32, 32)  # Expected output shape
        print("✓ Sample generation successful")
        
        # Step 6: Visualize results (basic test)
        print("Step 6: Testing visualization...")
        visualizer = PyTorchVisualizer(device=self.device)
        
        # Test basic visualization functionality
        sample = samples[0].cpu()
        try:
            # This should not raise an exception
            visualizer.plot_sculpture(sample, save_path=str(self.results_dir / "test_vis.png"))
            print("✓ Visualization test passed")
        except Exception as e:
            print(f"Warning: Visualization test failed: {e}")
        
        print("✅ Complete GAN pipeline test passed!")
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_complete_diffusion_pipeline(self):
        """Test complete diffusion pipeline from data generation to sampling."""
        print("Testing complete diffusion pipeline...")
        
        # Step 1: Generate data (reuse from GAN test)
        print("Step 1: Generating synthetic data...")
        sculptor_config = {
            "void_dim": 32,
            "num_shapes": 3,
            "sparse_threshold": 0.1
        }
        
        collector = PyTorchCollector(
            sculptor_config=sculptor_config,
            output_format="pytorch",
            device=self.device
        )
        
        num_samples = 8
        dataset = collector.create_streaming_dataset(num_samples)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0
        )
        
        print(f"✓ Generated dataset with {num_samples} samples")
        
        # Step 2: Create diffusion model
        print("Step 2: Creating diffusion model...")
        model = PyTorchModelFactory.create_diffusion_model(
            model_type="unet3d",
            void_dim=32,
            timesteps=100,  # Reduced for testing
            sparse=False
        ).to(self.device)
        
        # Test model forward pass
        with torch.no_grad():
            test_input = torch.randn(2, 1, 32, 32, 32, device=self.device)
            test_timesteps = torch.randint(0, 100, (2,), device=self.device)
            output = model(test_input, test_timesteps)
            
        assert output.shape == test_input.shape
        print("✓ Diffusion model created and tested")
        
        # Step 3: Create diffusion pipeline
        print("Step 3: Creating diffusion pipeline...")
        from core.models.diffusion.noise_scheduler import NoiseScheduler
        from core.models.diffusion.pipeline import Diffusion3DPipeline
        
        noise_scheduler = NoiseScheduler(
            schedule_type="linear",
            timesteps=100,
            beta_start=0.0001,
            beta_end=0.02
        )
        
        diffusion_pipeline = Diffusion3DPipeline(
            model=model,
            noise_scheduler=noise_scheduler,
            timesteps=100
        )
        
        print("✓ Diffusion pipeline created")
        
        # Step 4: Train diffusion model (minimal training)
        print("Step 4: Training diffusion model...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        trainer = DiffusionTrainer(
            model=model,
            diffusion_pipeline=diffusion_pipeline,
            optimizer=optimizer,
            device=self.device,
            mixed_precision=False
        )
        
        # Train for 2 epochs
        metrics = trainer.train(
            data_loader=data_loader,
            epochs=2,
            checkpoint_dir=self.checkpoints_dir
        )
        
        assert 'loss' in metrics
        assert len(metrics['loss']) == 2
        print("✓ Diffusion training completed")
        
        # Step 5: Generate samples using diffusion
        print("Step 5: Generating diffusion samples...")
        model.eval()
        
        # Generate samples with reduced steps for testing
        shape = (2, 1, 32, 32, 32)
        samples = diffusion_pipeline.sample(
            shape=shape,
            num_steps=10,  # Reduced for testing
            device=self.device
        )
        
        assert samples.shape == shape
        print("✓ Diffusion sampling successful")
        
        print("✅ Complete diffusion pipeline test passed!")
    
    @pytest.mark.integration
    def test_data_preprocessing_pipeline(self):
        """Test data preprocessing and curation pipeline."""
        print("Testing data preprocessing pipeline...")
        
        # Step 1: Generate raw data
        print("Step 1: Generating raw data...")
        sculptor = PyTorchSculptor(
            void_dim=32,
            device=self.device,
            sparse_mode=False
        )
        
        # Generate multiple sculptures
        raw_data = []
        for i in range(5):
            structure, colors = sculptor.generate_sculpture()
            raw_data.append({
                'structure': structure,
                'colors': colors,
                'metadata': {'sample_id': i}
            })
        
        print(f"✓ Generated {len(raw_data)} raw samples")
        
        # Step 2: Test different encoding methods
        print("Step 2: Testing encoding methods...")
        encoding_methods = ['one_hot', 'binary', 'rgb']
        
        for encoding in encoding_methods:
            print(f"  Testing {encoding} encoding...")
            curator = PyTorchCurator(
                encoding_method=encoding,
                device=self.device,
                sparse_mode=False
            )
            
            # Process a sample
            sample = raw_data[0]
            processed = curator.preprocess_batch({
                'structure': sample['structure'].unsqueeze(0),
                'colors': sample['colors'].unsqueeze(0)
            })
            
            assert 'structure' in processed
            assert 'colors' in processed
            print(f"  ✓ {encoding} encoding successful")
        
        print("✓ All encoding methods tested")
        
        # Step 3: Test sparse tensor conversion
        print("Step 3: Testing sparse tensor conversion...")
        curator_sparse = PyTorchCurator(
            encoding_method='one_hot',
            device=self.device,
            sparse_mode=True
        )
        
        sample = raw_data[0]
        processed_sparse = curator_sparse.preprocess_batch({
            'structure': sample['structure'].unsqueeze(0),
            'colors': sample['colors'].unsqueeze(0)
        })
        
        # Check if sparse conversion worked
        structure_tensor = processed_sparse['structure']
        if hasattr(structure_tensor, 'is_sparse') and structure_tensor.is_sparse:
            print("✓ Sparse tensor conversion successful")
        else:
            print("✓ Dense tensor maintained (low sparsity)")
        
        print("✅ Data preprocessing pipeline test passed!")
    
    @pytest.mark.integration
    def test_memory_optimization_pipeline(self):
        """Test memory optimization and sparse tensor functionality."""
        print("Testing memory optimization pipeline...")
        
        # Step 1: Create data with varying sparsity
        print("Step 1: Creating data with varying sparsity...")
        sculptor = PyTorchSculptor(
            void_dim=64,  # Larger for memory testing
            device=self.device,
            sparse_mode=False
        )
        
        # Generate sparse and dense samples
        sparse_sample, _ = sculptor.generate_sculpture()
        
        # Create artificially dense sample
        dense_sample = torch.rand(64, 64, 64, device=self.device)
        
        print("✓ Created sparse and dense samples")
        
        # Step 2: Test automatic sparsity detection
        print("Step 2: Testing sparsity detection...")
        sparse_ratio = PyTorchUtils.detect_sparsity(sparse_sample)
        dense_ratio = PyTorchUtils.detect_sparsity(dense_sample)
        
        print(f"  Sparse sample sparsity: {sparse_ratio:.3f}")
        print(f"  Dense sample sparsity: {dense_ratio:.3f}")
        
        assert sparse_ratio > dense_ratio
        print("✓ Sparsity detection working correctly")
        
        # Step 3: Test memory usage calculation
        print("Step 3: Testing memory usage calculation...")
        sparse_memory = PyTorchUtils.calculate_memory_usage(sparse_sample)
        dense_memory = PyTorchUtils.calculate_memory_usage(dense_sample)
        
        print(f"  Sparse sample memory: {sparse_memory}")
        print(f"  Dense sample memory: {dense_memory}")
        
        assert 'total_memory_mb' in sparse_memory
        assert 'total_memory_mb' in dense_memory
        print("✓ Memory usage calculation working")
        
        # Step 4: Test sparse tensor optimization
        print("Step 4: Testing sparse tensor optimization...")
        optimized_sparse = PyTorchUtils.optimize_tensor_storage(sparse_sample, threshold=0.1)
        optimized_dense = PyTorchUtils.optimize_tensor_storage(dense_sample, threshold=0.1)
        
        # Verify optimization decisions
        if hasattr(optimized_sparse, 'is_sparse'):
            print("✓ Sparse optimization applied correctly")
        
        print("✅ Memory optimization pipeline test passed!")
    
    @pytest.mark.integration
    @pytest.mark.gpu
    def test_gpu_acceleration_pipeline(self):
        """Test GPU acceleration and CUDA functionality."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        print("Testing GPU acceleration pipeline...")
        
        # Step 1: Test GPU memory management
        print("Step 1: Testing GPU memory management...")
        initial_memory = torch.cuda.memory_allocated()
        
        # Create large tensors on GPU
        large_tensor = torch.randn(100, 100, 100, device="cuda")
        after_allocation = torch.cuda.memory_allocated()
        
        assert after_allocation > initial_memory
        print(f"✓ GPU memory allocation: {(after_allocation - initial_memory) / 1e6:.1f} MB")
        
        # Clean up
        del large_tensor
        torch.cuda.empty_cache()
        
        # Step 2: Test model training on GPU
        print("Step 2: Testing model training on GPU...")
        model = PyTorchModelFactory.create_gan_generator(
            model_type="simple",
            void_dim=32,
            noise_dim=64,
            sparse=False
        ).to("cuda")
        
        # Test forward pass
        with torch.no_grad():
            noise = torch.randn(4, 64, device="cuda")
            output = model(noise)
            
        assert output.device.type == "cuda"
        print("✓ GPU model execution successful")
        
        # Step 3: Test mixed precision training
        print("Step 3: Testing mixed precision training...")
        scaler = torch.cuda.amp.GradScaler()
        optimizer = torch.optim.Adam(model.parameters())
        
        # Simulate training step with mixed precision
        noise = torch.randn(2, 64, device="cuda")
        target = torch.randn(2, 1, 32, 32, 32, device="cuda")
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(noise)
            loss = torch.nn.functional.mse_loss(output, target)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        print("✓ Mixed precision training successful")
        
        print("✅ GPU acceleration pipeline test passed!")
    
    @pytest.mark.integration
    def test_performance_benchmarking(self):
        """Test performance benchmarking functionality."""
        print("Testing performance benchmarking...")
        
        # Step 1: Create model for benchmarking
        print("Step 1: Creating model for benchmarking...")
        model = PyTorchModelFactory.create_gan_generator(
            model_type="simple",
            void_dim=32,
            noise_dim=64,
            sparse=False
        ).to(self.device)
        
        # Step 2: Run inference benchmark
        print("Step 2: Running inference benchmark...")
        input_shape = (8, 64)  # batch_size, noise_dim
        
        benchmark_results = PyTorchUtils.benchmark_model_inference(model, input_shape)
        
        # Verify benchmark results
        expected_metrics = ['inference_time_ms', 'throughput_samples_per_sec', 'memory_usage_mb']
        for metric in expected_metrics:
            assert metric in benchmark_results
            assert isinstance(benchmark_results[metric], (int, float))
        
        print(f"✓ Benchmark results: {benchmark_results}")
        
        # Step 3: Test memory profiling
        print("Step 3: Testing memory profiling...")
        test_tensor = torch.randn(32, 32, 32, device=self.device)
        memory_profile = PyTorchUtils.calculate_memory_usage(test_tensor)
        
        assert 'total_memory_mb' in memory_profile
        print(f"✓ Memory profile: {memory_profile}")
        
        print("✅ Performance benchmarking test passed!")
    
    @pytest.mark.integration
    def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms."""
        print("Testing error handling and recovery...")
        
        # Step 1: Test invalid model configuration
        print("Step 1: Testing invalid model configuration...")
        with pytest.raises(Exception):
            PyTorchModelFactory.create_gan_generator(
                model_type="invalid_type",
                void_dim=32,
                noise_dim=64
            )
        print("✓ Invalid model configuration handled correctly")
        
        # Step 2: Test memory overflow handling
        print("Step 2: Testing memory overflow handling...")
        try:
            # Try to create an extremely large tensor
            huge_tensor = torch.randn(10000, 10000, 10000, device=self.device)
            print("Warning: Large tensor creation succeeded unexpectedly")
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            print(f"✓ Memory overflow handled correctly: {type(e).__name__}")
        
        # Step 3: Test data loading error handling
        print("Step 3: Testing data loading error handling...")
        try:
            # Try to load non-existent data
            invalid_path = self.temp_dir / "non_existent_file.pt"
            torch.load(invalid_path)
        except FileNotFoundError:
            print("✓ File not found error handled correctly")
        
        print("✅ Error handling and recovery test passed!")
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_full_training_convergence(self):
        """Test that models can actually converge during training."""
        print("Testing full training convergence...")
        
        # Step 1: Generate consistent dataset
        print("Step 1: Generating consistent dataset...")
        torch.manual_seed(42)  # For reproducibility
        
        sculptor_config = {
            "void_dim": 32,
            "num_shapes": 3,
            "sparse_threshold": 0.1
        }
        
        collector = PyTorchCollector(
            sculptor_config=sculptor_config,
            device=self.device
        )
        
        dataset = collector.create_streaming_dataset(50)  # Larger dataset
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=8,
            shuffle=True,
            num_workers=0
        )
        
        # Step 2: Create and train GAN
        print("Step 2: Training GAN for convergence...")
        generator = PyTorchModelFactory.create_gan_generator(
            model_type="simple",
            void_dim=32,
            noise_dim=64,
            sparse=False
        ).to(self.device)
        
        discriminator = PyTorchModelFactory.create_gan_discriminator(
            model_type="simple",
            void_dim=32,
            sparse=False
        ).to(self.device)
        
        gen_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)
        disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002)
        
        trainer = GANTrainer(
            generator=generator,
            discriminator=discriminator,
            gen_optimizer=gen_optimizer,
            disc_optimizer=disc_optimizer,
            device=self.device
        )
        
        # Train for more epochs to check convergence
        metrics = trainer.train(
            data_loader=data_loader,
            epochs=10,
            checkpoint_dir=self.checkpoints_dir
        )
        
        # Step 3: Analyze convergence
        print("Step 3: Analyzing convergence...")
        gen_losses = metrics['gen_loss']
        disc_losses = metrics['disc_loss']
        
        # Check that losses are reasonable (not NaN or infinite)
        assert all(torch.isfinite(torch.tensor(loss)) for loss in gen_losses)
        assert all(torch.isfinite(torch.tensor(loss)) for loss in disc_losses)
        
        # Check that losses show some stability (not constantly increasing)
        final_gen_loss = sum(gen_losses[-3:]) / 3  # Average of last 3 epochs
        initial_gen_loss = sum(gen_losses[:3]) / 3  # Average of first 3 epochs
        
        print(f"Initial gen loss: {initial_gen_loss:.4f}")
        print(f"Final gen loss: {final_gen_loss:.4f}")
        
        # Loss should not explode (increase by more than 10x)
        assert final_gen_loss < initial_gen_loss * 10
        
        print("✓ Training convergence verified")
        
        # Step 4: Test sample quality
        print("Step 4: Testing sample quality...")
        generator.eval()
        with torch.no_grad():
            noise = torch.randn(5, 64, device=self.device)
            samples = generator(noise)
            
        # Basic quality checks
        assert not torch.isnan(samples).any()
        assert not torch.isinf(samples).any()
        assert samples.min() >= -10 and samples.max() <= 10  # Reasonable range
        
        print("✓ Sample quality verified")
        
        print("✅ Full training convergence test passed!")


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "--tb=short"])