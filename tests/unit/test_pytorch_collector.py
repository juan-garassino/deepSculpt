"""
Comprehensive tests for PyTorchCollector dataset generation and streaming.
Tests cover streaming dataset creation, memory optimization, distributed processing,
and various output formats.
"""

import os
import sys
import tempfile
import shutil
import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the deepSculpt module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'deepSculpt'))

from deepsculpt.core.data.generation.pytorch_collector import (
    PyTorchCollector,
    StreamingDataset,
    MemoryMonitor,
    DistributedCollector,
    PyTorchCollectorDistributed,
    load_pytorch_sample,
    load_numpy_sample,
    list_available_collections,
)


class TestMemoryMonitor:
    """Test memory monitoring functionality."""
    
    def test_memory_monitor_initialization(self):
        """Test MemoryMonitor initialization."""
        monitor = MemoryMonitor(memory_limit_gb=4.0, safety_margin=0.8)
        
        assert monitor.memory_limit_gb == 4.0
        assert monitor.safety_margin == 0.8
        assert monitor.safe_limit_gb == 3.2
    
    def test_get_memory_usage(self):
        """Test memory usage reporting."""
        monitor = MemoryMonitor()
        usage = monitor.get_memory_usage()
        
        # Check that all expected keys are present
        expected_keys = [
            "gpu_allocated", "gpu_reserved", "gpu_max",
            "cpu_memory", "system_available", "system_percent"
        ]
        
        for key in expected_keys:
            assert key in usage
            assert isinstance(usage[key], float)
            assert usage[key] >= 0
    
    def test_suggest_batch_size(self):
        """Test batch size suggestion."""
        monitor = MemoryMonitor(memory_limit_gb=4.0)
        
        # Test with reasonable sample size
        suggested = monitor.suggest_batch_size(current_batch_size=32, sample_memory_mb=100.0)
        assert isinstance(suggested, int)
        assert suggested > 0
        
        # Test with very large sample size
        suggested_small = monitor.suggest_batch_size(current_batch_size=32, sample_memory_mb=5000.0)
        assert suggested_small <= suggested


class TestStreamingDataset:
    """Test streaming dataset functionality."""
    
    @pytest.fixture
    def sculptor_config(self):
        """Basic sculptor configuration for testing."""
        return {
            "void_dim": 16,  # Small for fast testing
            "edges": (1, 0.3, 0.5),
            "planes": (0, 0.3, 0.5),  # Disable for speed
            "pipes": (0, 0.3, 0.5),   # Disable for speed
            "grid": (0, 4),           # Disable for speed
            "step": 1,
        }
    
    def test_streaming_dataset_initialization(self, sculptor_config):
        """Test StreamingDataset initialization."""
        dataset = StreamingDataset(
            sculptor_config=sculptor_config,
            num_samples=10,
            device="cpu",
            sparse_mode=False,
            cache_size=5,
        )
        
        assert len(dataset) == 10
        assert dataset.device == "cpu"
        assert dataset.cache_size == 5
        assert len(dataset._cache) == 0
    
    def test_streaming_dataset_sample_generation(self, sculptor_config):
        """Test sample generation from streaming dataset."""
        dataset = StreamingDataset(
            sculptor_config=sculptor_config,
            num_samples=5,
            device="cpu",
            cache_size=2,
        )
        
        # Generate first sample
        sample = dataset[0]
        
        assert "structure" in sample
        assert "colors" in sample
        assert "index" in sample
        
        assert isinstance(sample["structure"], torch.Tensor)
        assert isinstance(sample["colors"], torch.Tensor)
        assert sample["index"].item() == 0
        
        # Check tensor shapes
        expected_shape = (16, 16, 16)  # void_dim from config
        assert sample["structure"].shape == expected_shape
        assert sample["colors"].shape == expected_shape
    
    def test_streaming_dataset_caching(self, sculptor_config):
        """Test caching behavior."""
        dataset = StreamingDataset(
            sculptor_config=sculptor_config,
            num_samples=5,
            device="cpu",
            cache_size=2,
        )
        
        # Generate samples to fill cache
        sample1 = dataset[0]
        sample2 = dataset[1]
        
        # Check cache
        assert len(dataset._cache) == 2
        assert 0 in dataset._cache
        assert 1 in dataset._cache
        
        # Generate another sample (should evict oldest)
        sample3 = dataset[2]
        
        assert len(dataset._cache) == 2
        assert 0 not in dataset._cache  # Should be evicted
        assert 1 in dataset._cache
        assert 2 in dataset._cache
    
    def test_streaming_dataset_cache_clear(self, sculptor_config):
        """Test cache clearing."""
        dataset = StreamingDataset(
            sculptor_config=sculptor_config,
            num_samples=5,
            device="cpu",
            cache_size=3,
        )
        
        # Fill cache
        for i in range(3):
            _ = dataset[i]
        
        assert len(dataset._cache) == 3
        
        # Clear cache
        dataset.clear_cache()
        
        assert len(dataset._cache) == 0
        assert len(dataset._cache_order) == 0


class TestPyTorchCollector:
    """Test PyTorchCollector functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sculptor_config(self):
        """Basic sculptor configuration for testing."""
        return {
            "void_dim": 16,  # Small for fast testing
            "edges": (1, 0.3, 0.5),
            "planes": (0, 0.3, 0.5),
            "pipes": (0, 0.3, 0.5),
            "grid": (0, 4),
            "step": 1,
        }
    
    def test_collector_initialization(self, temp_dir, sculptor_config):
        """Test PyTorchCollector initialization."""
        collector = PyTorchCollector(
            sculptor_config=sculptor_config,
            output_format="pytorch",
            base_dir=temp_dir,
            device="cpu",
            verbose=False,
        )
        
        assert collector.output_format == "pytorch"
        assert collector.device == "cpu"
        assert collector.base_dir == Path(temp_dir)
        assert collector.sculptor_config == sculptor_config
    
    def test_collector_invalid_format(self, temp_dir, sculptor_config):
        """Test initialization with invalid output format."""
        with pytest.raises(ValueError, match="output_format must be one of"):
            PyTorchCollector(
                sculptor_config=sculptor_config,
                output_format="invalid_format",
                base_dir=temp_dir,
            )
    
    def test_create_streaming_dataset(self, temp_dir, sculptor_config):
        """Test streaming dataset creation."""
        collector = PyTorchCollector(
            sculptor_config=sculptor_config,
            base_dir=temp_dir,
            device="cpu",
            verbose=False,
        )
        
        dataset = collector.create_streaming_dataset(
            num_samples=10,
            cache_size=5,
        )
        
        assert isinstance(dataset, StreamingDataset)
        assert len(dataset) == 10
        assert dataset.cache_size == 5
    
    def test_create_collection_pytorch_format(self, temp_dir, sculptor_config):
        """Test collection creation with PyTorch format."""
        collector = PyTorchCollector(
            sculptor_config=sculptor_config,
            output_format="pytorch",
            base_dir=temp_dir,
            device="cpu",
            verbose=False,
        )
        
        sample_paths = collector.create_collection(
            num_samples=3,
            batch_size=2,
            dynamic_batching=False,
        )
        
        assert len(sample_paths) == 3
        
        # Check that files were created
        for path in sample_paths:
            assert os.path.exists(path)
            assert path.endswith('.pt')
        
        # Check directory structure
        assert collector.structures_dir.exists()
        assert collector.colors_dir.exists()
        assert collector.metadata_dir.exists()
        
        # Check metadata file
        metadata_file = collector.metadata_dir / "collection_metadata.json"
        assert metadata_file.exists()
    
    def test_create_collection_numpy_format(self, temp_dir, sculptor_config):
        """Test collection creation with NumPy format."""
        collector = PyTorchCollector(
            sculptor_config=sculptor_config,
            output_format="numpy",
            base_dir=temp_dir,
            device="cpu",
            verbose=False,
        )
        
        sample_paths = collector.create_collection(
            num_samples=2,
            batch_size=1,
        )
        
        assert len(sample_paths) == 2
        
        # Check that files were created
        for path in sample_paths:
            assert os.path.exists(path)
            assert path.endswith('.npy')
    
    def test_memory_usage_estimation(self, temp_dir, sculptor_config):
        """Test memory usage estimation."""
        collector = PyTorchCollector(
            sculptor_config=sculptor_config,
            base_dir=temp_dir,
            device="cpu",
        )
        
        estimates = collector.estimate_memory_usage(void_dim=32, batch_size=4)
        
        expected_keys = [
            "structure_mb", "colors_mb", "sample_mb",
            "batch_mb", "total_mb", "recommended_batch_size"
        ]
        
        for key in expected_keys:
            assert key in estimates
            assert isinstance(estimates[key], (int, float))
            assert estimates[key] > 0
    
    def test_generation_stats(self, temp_dir, sculptor_config):
        """Test generation statistics tracking."""
        collector = PyTorchCollector(
            sculptor_config=sculptor_config,
            base_dir=temp_dir,
            device="cpu",
            verbose=False,
        )
        
        # Initially empty stats
        stats = collector.get_generation_stats()
        assert stats["total_samples"] == 0
        
        # Generate some samples
        collector.create_collection(num_samples=2, batch_size=1)
        
        # Check updated stats
        stats = collector.get_generation_stats()
        assert stats["total_samples"] == 2
        assert stats["generation_time"] > 0
        assert len(stats["batch_sizes"]) > 0


class TestDistributedCollector:
    """Test distributed collection functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def collector_config(self, temp_dir):
        """Basic collector configuration for testing."""
        return {
            "sculptor_config": {
                "void_dim": 16,
                "edges": (1, 0.3, 0.5),
                "planes": (0, 0.3, 0.5),
                "pipes": (0, 0.3, 0.5),
                "grid": (0, 4),
                "step": 1,
            },
            "output_format": "pytorch",
            "base_dir": temp_dir,
            "device": "cpu",
            "verbose": False,
        }
    
    def test_distributed_collector_initialization(self, collector_config):
        """Test DistributedCollector initialization."""
        distributed_collector = DistributedCollector(
            collector_config=collector_config,
            num_gpus=1,  # Force single GPU for testing
            verbose=False,
        )
        
        assert distributed_collector.num_gpus == 1
        assert len(distributed_collector.devices) == 1
        assert distributed_collector.fault_tolerance == True
    
    def test_work_distribution(self, collector_config):
        """Test work distribution across workers."""
        distributed_collector = DistributedCollector(
            collector_config=collector_config,
            num_gpus=2,
            verbose=False,
        )
        
        # Override devices for testing
        distributed_collector.devices = ["cpu", "cpu"]
        
        work_assignments = distributed_collector.distribute_work(
            num_samples=10,
            batch_size=2,
        )
        
        assert len(work_assignments) == 2
        
        # Check work distribution
        total_assigned = sum(assignment["num_samples"] for assignment in work_assignments.values())
        assert total_assigned == 10
        
        # Check assignment structure
        for worker_id, assignment in work_assignments.items():
            assert "device" in assignment
            assert "start_idx" in assignment
            assert "num_samples" in assignment
            assert "batch_size" in assignment
            assert "worker_id" in assignment
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_single_process_fallback(self, mock_cuda, collector_config):
        """Test fallback to single process when CUDA unavailable."""
        distributed_collector = DistributedCollector(
            collector_config=collector_config,
            verbose=False,
        )
        
        # Should fall back to single CPU process
        assert distributed_collector.num_gpus == 1
        assert distributed_collector.devices == ["cpu"]
    
    def test_distributed_stats(self, collector_config):
        """Test distributed statistics tracking."""
        distributed_collector = DistributedCollector(
            collector_config=collector_config,
            num_gpus=1,
            verbose=False,
        )
        
        stats = distributed_collector.get_distributed_stats()
        
        expected_keys = [
            "num_workers", "devices", "work_assignments",
            "completed_work", "failed_work", "total_samples", "completed_samples"
        ]
        
        for key in expected_keys:
            assert key in stats


class TestUtilityFunctions:
    """Test utility functions for loading samples."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_load_pytorch_sample(self, temp_dir):
        """Test loading PyTorch samples."""
        # Create test tensors
        structure = torch.randint(0, 2, (8, 8, 8), dtype=torch.int8)
        colors = torch.randint(0, 10, (8, 8, 8), dtype=torch.int16)
        
        # Save tensors
        structure_path = os.path.join(temp_dir, "structure.pt")
        colors_path = os.path.join(temp_dir, "colors.pt")
        
        torch.save(structure, structure_path)
        torch.save(colors, colors_path)
        
        # Load and verify
        loaded_structure, loaded_colors = load_pytorch_sample(structure_path, colors_path)
        
        assert torch.equal(structure, loaded_structure)
        assert torch.equal(colors, loaded_colors)
    
    def test_load_numpy_sample(self, temp_dir):
        """Test loading NumPy samples."""
        # Create test arrays
        structure = np.random.randint(0, 2, (8, 8, 8), dtype=np.int8)
        colors = np.random.randint(0, 10, (8, 8, 8), dtype=np.int16)
        
        # Save arrays
        structure_path = os.path.join(temp_dir, "structure.npy")
        colors_path = os.path.join(temp_dir, "colors.npy")
        
        np.save(structure_path, structure)
        np.save(colors_path, colors)
        
        # Load and verify
        loaded_structure, loaded_colors = load_numpy_sample(structure_path, colors_path)
        
        assert np.array_equal(structure, loaded_structure)
        assert np.array_equal(colors, loaded_colors)
    
    def test_list_available_collections_empty(self, temp_dir):
        """Test listing collections from empty directory."""
        collections = list_available_collections(temp_dir)
        assert collections == []
    
    def test_list_available_collections_nonexistent(self):
        """Test listing collections from nonexistent directory."""
        collections = list_available_collections("/nonexistent/path")
        assert collections == []


class TestIntegration:
    """Integration tests for complete workflows."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_end_to_end_pytorch_workflow(self, temp_dir):
        """Test complete PyTorch workflow from generation to loading."""
        sculptor_config = {
            "void_dim": 12,
            "edges": (1, 0.3, 0.5),
            "planes": (0, 0.3, 0.5),
            "pipes": (0, 0.3, 0.5),
            "grid": (0, 4),
        }
        
        # Create collector
        collector = PyTorchCollector(
            sculptor_config=sculptor_config,
            output_format="pytorch",
            base_dir=temp_dir,
            device="cpu",
            verbose=False,
        )
        
        # Generate collection
        sample_paths = collector.create_collection(
            num_samples=3,
            batch_size=2,
        )
        
        assert len(sample_paths) == 3
        
        # Test loading samples
        for i, structure_path in enumerate(sample_paths):
            # Construct the correct colors path
            structure_file = Path(structure_path).name
            colors_file = structure_file.replace("structure_", "colors_")
            colors_path = str(Path(structure_path).parent.parent / "colors" / colors_file)
            
            structure, colors = load_pytorch_sample(structure_path, colors_path)
            
            assert isinstance(structure, torch.Tensor)
            assert isinstance(colors, torch.Tensor)
            assert structure.shape == (12, 12, 12)
            assert colors.shape == (12, 12, 12)
        
        # Test streaming dataset
        streaming_dataset = collector.create_streaming_dataset(
            num_samples=5,
            cache_size=2,
        )
        
        # Test dataset access
        sample = streaming_dataset[0]
        assert "structure" in sample
        assert "colors" in sample
        assert sample["structure"].shape == (12, 12, 12)
        
        # Test collection listing
        collections = list_available_collections(temp_dir)
        assert len(collections) >= 1
    
    def test_memory_optimization_workflow(self, temp_dir):
        """Test memory optimization features."""
        sculptor_config = {
            "void_dim": 16,
            "edges": (1, 0.3, 0.5),
            "planes": (0, 0.3, 0.5),
            "pipes": (0, 0.3, 0.5),
            "grid": (0, 4),
        }
        
        # Create collector with small memory limit
        collector = PyTorchCollector(
            sculptor_config=sculptor_config,
            base_dir=temp_dir,
            device="cpu",
            memory_limit_gb=0.1,  # Very small limit to trigger optimization
            sparse_mode=True,
            verbose=False,
        )
        
        # Test memory estimation
        estimates = collector.estimate_memory_usage(void_dim=16, batch_size=4)
        assert estimates["recommended_batch_size"] >= 1
        
        # Generate with dynamic batching
        sample_paths = collector.create_collection(
            num_samples=4,
            batch_size=8,  # Start with large batch
            dynamic_batching=True,
        )
        
        assert len(sample_paths) == 4
        
        # Check that batch sizes were adjusted
        stats = collector.get_generation_stats()
        assert len(stats["batch_sizes"]) > 0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])