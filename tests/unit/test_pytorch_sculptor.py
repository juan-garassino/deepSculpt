"""
Comprehensive tests for PyTorchSculptor class.
Tests sculpture generation equivalence, memory optimization, device management,
and all manipulation methods.
"""

import pytest
import torch
import numpy as np
import tempfile
import os
import shutil
from typing import Tuple, Dict, Any

# Import the modules to test
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'deepSculpt'))

from deepsculpt.core.data.generation.pytorch_sculptor import PyTorchSculptor, create_pytorch_sculptor
from deepsculpt.core.data.generation.pytorch_shapes import ShapeType, SparseTensorHandler
from sculptor import Sculptor  # Original implementation for comparison


class TestPyTorchSculptorCore:
    """Test core functionality of PyTorchSculptor."""
    
    def test_initialization_default(self):
        """Test default initialization."""
        sculptor = PyTorchSculptor()
        
        assert sculptor.void_dim == 20
        assert sculptor.device in ["cpu", "cuda"]
        assert sculptor.structure.shape == (20, 20, 20)
        assert sculptor.colors.shape == (20, 20, 20)
        assert sculptor.structure.dtype == torch.int8
        assert sculptor.colors.dtype == torch.int16
    
    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        sculptor = PyTorchSculptor(
            void_dim=32,
            edges=(2, 0.2, 0.6),
            planes=(1, 0.3, 0.7),
            pipes=(1, 0.4, 0.8),
            grid=(1, 3),
            device="cpu",
            sparse_mode=True,
            sparse_threshold=0.2,
            memory_limit_gb=2.0,
        )
        
        assert sculptor.void_dim == 32
        assert sculptor.edges == (2, 0.2, 0.6)
        assert sculptor.planes == (1, 0.3, 0.7)
        assert sculptor.pipes == (1, 0.4, 0.8)
        assert sculptor.grid == (1, 3)
        assert sculptor.device == "cpu"
        assert sculptor.sparse_mode == True
        assert sculptor.sparse_threshold == 0.2
        assert sculptor.memory_limit_gb == 2.0
    
    def test_initialization_validation(self):
        """Test parameter validation during initialization."""
        # Test invalid void_dim
        with pytest.raises(ValueError, match="void_dim must be positive"):
            PyTorchSculptor(void_dim=0)
        
        # Test invalid edges parameters
        with pytest.raises(ValueError, match="edges must be a tuple"):
            PyTorchSculptor(edges=(1, 0.5))  # Missing third parameter
        
        with pytest.raises(ValueError, match="edges ratios must be in"):
            PyTorchSculptor(edges=(1, 0.8, 0.5))  # min > max
        
        # Test invalid step
        with pytest.raises(ValueError, match="step must be positive"):
            PyTorchSculptor(step=0)
        
        # Test invalid sparse_threshold
        with pytest.raises(ValueError, match="sparse_threshold must be in"):
            PyTorchSculptor(sparse_threshold=1.5)
    
    def test_device_management(self):
        """Test device management functionality."""
        sculptor = PyTorchSculptor(device="cpu")
        assert sculptor.device == "cpu"
        assert sculptor.structure.device.type == "cpu"
        assert sculptor.colors.device.type == "cpu"
        
        # Test device switching
        if torch.cuda.is_available():
            sculptor.to_device("cuda")
            assert sculptor.device == "cuda"
            assert sculptor.structure.device.type == "cuda"
            assert sculptor.colors.device.type == "cuda"
            
            sculptor.to_device("cpu")
            assert sculptor.device == "cpu"
            assert sculptor.structure.device.type == "cpu"
            assert sculptor.colors.device.type == "cpu"
    
    def test_sparse_dense_conversion(self):
        """Test sparse/dense tensor conversion."""
        sculptor = PyTorchSculptor(void_dim=16, sparse_mode=False)
        
        # Initially dense
        assert not sculptor.structure.is_sparse
        assert not sculptor.colors.is_sparse
        
        # Add some data to make it sparse-worthy
        sculptor.structure[0:8, 0:8, 0:8] = 1
        sculptor.colors[0:8, 0:8, 0:8] = 1
        
        # Convert to sparse
        sculptor.to_sparse()
        # Note: PyTorch may not convert to sparse if not beneficial
        
        # Convert back to dense
        sculptor.to_dense()
        assert not sculptor.structure.is_sparse
        assert not sculptor.colors.is_sparse
    
    def test_memory_usage_tracking(self):
        """Test memory usage tracking."""
        sculptor = PyTorchSculptor(void_dim=16)
        
        memory_usage = sculptor.get_memory_usage()
        assert "allocated" in memory_usage
        assert "reserved" in memory_usage
        assert "max_allocated" in memory_usage
        assert all(isinstance(v, float) for v in memory_usage.values())
    
    def test_tensor_info(self):
        """Test tensor information retrieval."""
        sculptor = PyTorchSculptor(void_dim=16)
        
        info = sculptor.get_tensor_info()
        assert "structure" in info
        assert "colors" in info
        assert "sparsity" in info
        assert "memory_usage" in info
        assert "sparse_mode" in info
        
        # Check structure info
        struct_info = info["structure"]
        assert struct_info["shape"] == (16, 16, 16)
        assert "int8" in struct_info["dtype"]
        assert struct_info["is_sparse"] == False
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        sculptor = PyTorchSculptor(void_dim=16)
        assert sculptor.validate_configuration() == True
        
        # Test with mismatched tensor shapes (artificially created)
        sculptor.colors = torch.zeros((8, 8, 8), dtype=torch.int16, device=sculptor.device)
        assert sculptor.validate_configuration() == False


class TestPyTorchSculptorGeneration:
    """Test sculpture generation functionality."""
    
    def test_generate_sculpture_basic(self):
        """Test basic sculpture generation."""
        sculptor = PyTorchSculptor(
            void_dim=16,
            edges=(1, 0.2, 0.4),
            planes=(1, 0.2, 0.4),
            pipes=(1, 0.2, 0.4),
            grid=(1, 4),
        )
        
        structure, colors = sculptor.generate_sculpture()
        
        # Check that sculpture was generated
        assert structure.shape == (16, 16, 16)
        assert colors.shape == (16, 16, 16)
        
        # Check that some voxels are filled
        filled_voxels = torch.sum(structure > 0).item()
        assert filled_voxels > 0
        
        # Check generation stats
        stats = sculptor.get_generation_stats()
        assert "generation_time" in stats
        assert "filled_voxels" in stats
        assert "fill_percentage" in stats
        assert stats["filled_voxels"] == filled_voxels
    
    def test_generate_sculpture_empty(self):
        """Test sculpture generation with no components."""
        sculptor = PyTorchSculptor(
            void_dim=16,
            edges=(0, 0.2, 0.4),
            planes=(0, 0.2, 0.4),
            pipes=(0, 0.2, 0.4),
            grid=(0, 4),
        )
        
        structure, colors = sculptor.generate_sculpture()
        
        # Should be empty
        filled_voxels = torch.sum(structure > 0).item()
        assert filled_voxels == 0
    
    def test_generate_sculpture_with_history(self):
        """Test sculpture generation with history saving."""
        sculptor = PyTorchSculptor(void_dim=16, edges=(1, 0.2, 0.4))
        
        # Generate with history
        sculptor.generate_sculpture(save_to_history=True)
        
        history_info = sculptor.get_history_info()
        assert history_info["history_size"] > 0
        assert history_info["can_undo"] == True
    
    def test_sculpture_quality_validation(self):
        """Test sculpture quality validation."""
        sculptor = PyTorchSculptor(void_dim=16, edges=(1, 0.2, 0.4))
        sculptor.generate_sculpture()
        
        # Should pass validation
        assert sculptor._validate_sculpture_quality() == True
    
    def test_generation_time_estimation(self):
        """Test generation time estimation."""
        sculptor = PyTorchSculptor(
            void_dim=16,
            edges=(2, 0.2, 0.4),
            planes=(1, 0.2, 0.4),
            pipes=(1, 0.2, 0.4),
            grid=(1, 4),
        )
        
        estimated_time = sculptor.estimate_generation_time()
        assert isinstance(estimated_time, float)
        assert estimated_time > 0


class TestPyTorchSculptorManipulation:
    """Test sculpture manipulation methods."""
    
    def test_add_shape(self):
        """Test adding individual shapes."""
        sculptor = PyTorchSculptor(void_dim=16)
        
        # Add an edge
        sculptor.add_shape(ShapeType.EDGE, 0.2, 0.4)
        filled_voxels_1 = torch.sum(sculptor.structure > 0).item()
        assert filled_voxels_1 > 0
        
        # Add a plane
        sculptor.add_shape(ShapeType.PLANE, 0.2, 0.4)
        filled_voxels_2 = torch.sum(sculptor.structure > 0).item()
        assert filled_voxels_2 >= filled_voxels_1
        
        # Add a pipe
        sculptor.add_shape(ShapeType.PIPE, 0.2, 0.4)
        filled_voxels_3 = torch.sum(sculptor.structure > 0).item()
        assert filled_voxels_3 >= filled_voxels_2
    
    def test_method_chaining(self):
        """Test method chaining functionality."""
        sculptor = PyTorchSculptor(void_dim=16)
        
        # Test chaining
        result = (sculptor
                 .add_shape(ShapeType.EDGE, 0.2, 0.4)
                 .add_shape(ShapeType.PLANE, 0.2, 0.4)
                 .to_sparse()
                 .to_dense())
        
        assert result is sculptor  # Should return self
        assert torch.sum(sculptor.structure > 0).item() > 0
    
    def test_reset(self):
        """Test sculpture reset functionality."""
        sculptor = PyTorchSculptor(void_dim=16)
        
        # Add some shapes
        sculptor.add_shape(ShapeType.EDGE, 0.2, 0.4)
        assert torch.sum(sculptor.structure > 0).item() > 0
        
        # Reset
        sculptor.reset()
        assert torch.sum(sculptor.structure > 0).item() == 0
        assert torch.sum(sculptor.colors > 0).item() == 0
    
    def test_undo_redo(self):
        """Test undo/redo functionality."""
        sculptor = PyTorchSculptor(void_dim=16)
        
        # Initial state
        initial_filled = torch.sum(sculptor.structure > 0).item()
        
        # Add a shape
        sculptor.add_shape(ShapeType.EDGE, 0.2, 0.4, save_to_history=True)
        after_add_filled = torch.sum(sculptor.structure > 0).item()
        assert after_add_filled > initial_filled
        
        # Undo
        sculptor.undo()
        after_undo_filled = torch.sum(sculptor.structure > 0).item()
        assert after_undo_filled == initial_filled
        
        # Redo
        sculptor.redo()
        after_redo_filled = torch.sum(sculptor.structure > 0).item()
        assert after_redo_filled == after_add_filled
    
    def test_history_management(self):
        """Test history management."""
        sculptor = PyTorchSculptor(void_dim=16)
        
        # Check initial history
        history_info = sculptor.get_history_info()
        assert history_info["history_size"] == 0
        assert history_info["can_undo"] == False
        assert history_info["can_redo"] == False
        
        # Add operations to history
        for i in range(3):
            sculptor.add_shape(ShapeType.EDGE, 0.1, 0.2, save_to_history=True)
        
        history_info = sculptor.get_history_info()
        assert history_info["history_size"] > 0
        assert history_info["can_undo"] == True
        
        # Clear history
        sculptor.clear_history()
        history_info = sculptor.get_history_info()
        assert history_info["history_size"] == 0
    
    def test_transformation_translation(self):
        """Test translation transformation."""
        sculptor = PyTorchSculptor(void_dim=16)
        sculptor.add_shape(ShapeType.EDGE, 0.2, 0.4)
        
        original_structure = sculptor.structure.clone()
        
        # Apply translation
        sculptor.transform(translation=(2, 2, 2))
        
        # Structure should be different
        assert not torch.equal(sculptor.structure, original_structure)
    
    def test_transformation_scaling(self):
        """Test scaling transformation."""
        sculptor = PyTorchSculptor(void_dim=16)
        sculptor.add_shape(ShapeType.EDGE, 0.2, 0.4)
        
        original_structure = sculptor.structure.clone()
        
        # Apply scaling
        sculptor.transform(scale=0.5)
        
        # Structure should be different
        assert not torch.equal(sculptor.structure, original_structure)
    
    def test_merge_operations(self):
        """Test sculpture merging operations."""
        sculptor1 = PyTorchSculptor(void_dim=16)
        sculptor2 = PyTorchSculptor(void_dim=16)
        
        # Add different shapes to each
        sculptor1.add_shape(ShapeType.EDGE, 0.2, 0.4)
        sculptor2.add_shape(ShapeType.PLANE, 0.2, 0.4)
        
        filled_1 = torch.sum(sculptor1.structure > 0).item()
        filled_2 = torch.sum(sculptor2.structure > 0).item()
        
        # Test union
        sculptor1_copy = sculptor1.clone()
        sculptor1_copy.merge(sculptor2, operation="union")
        filled_union = torch.sum(sculptor1_copy.structure > 0).item()
        assert filled_union >= max(filled_1, filled_2)
        
        # Test intersection (might be 0 if no overlap)
        sculptor1_copy = sculptor1.clone()
        sculptor1_copy.merge(sculptor2, operation="intersection")
        filled_intersection = torch.sum(sculptor1_copy.structure > 0).item()
        assert filled_intersection >= 0
        
        # Test difference
        sculptor1_copy = sculptor1.clone()
        sculptor1_copy.merge(sculptor2, operation="difference")
        filled_difference = torch.sum(sculptor1_copy.structure > 0).item()
        assert filled_difference <= filled_1


class TestPyTorchSculptorIO:
    """Test save/load functionality."""
    
    def setUp(self):
        """Set up temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary directory."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_save_load_basic(self):
        """Test basic save and load functionality."""
        self.setUp()
        
        try:
            # Create and generate sculpture
            sculptor = PyTorchSculptor(void_dim=16, edges=(1, 0.2, 0.4))
            sculptor.generate_sculpture()
            
            # Save
            saved_files = sculptor.save(
                directory=self.temp_dir,
                filename_prefix="test_sculpture",
            )
            
            assert "structure" in saved_files
            assert "colors" in saved_files
            assert "metadata" in saved_files
            
            # Check files exist
            for file_path in saved_files.values():
                assert os.path.exists(file_path)
            
            # Load
            loaded_sculptor = PyTorchSculptor.load(
                structure_path=saved_files["structure"],
                colors_path=saved_files["colors"],
                metadata_path=saved_files["metadata"],
            )
            
            # Compare
            assert loaded_sculptor.void_dim == sculptor.void_dim
            assert torch.equal(loaded_sculptor.structure, sculptor.structure)
            assert torch.equal(loaded_sculptor.colors, sculptor.colors)
            
        finally:
            self.tearDown()
    
    def test_save_load_without_metadata(self):
        """Test save and load without metadata."""
        self.setUp()
        
        try:
            sculptor = PyTorchSculptor(void_dim=16, edges=(1, 0.2, 0.4))
            sculptor.generate_sculpture()
            
            # Save without metadata
            saved_files = sculptor.save(
                directory=self.temp_dir,
                save_metadata=False,
            )
            
            assert "metadata" not in saved_files
            
            # Load without metadata
            loaded_sculptor = PyTorchSculptor.load(
                structure_path=saved_files["structure"],
                colors_path=saved_files["colors"],
            )
            
            # Should have default parameters but same tensors
            assert loaded_sculptor.void_dim == sculptor.void_dim
            assert torch.equal(loaded_sculptor.structure, sculptor.structure)
            
        finally:
            self.tearDown()
    
    def test_clone(self):
        """Test sculpture cloning."""
        sculptor = PyTorchSculptor(void_dim=16, edges=(1, 0.2, 0.4))
        sculptor.generate_sculpture()
        
        # Clone
        cloned_sculptor = sculptor.clone()
        
        # Should be equal but different objects
        assert cloned_sculptor is not sculptor
        assert cloned_sculptor.void_dim == sculptor.void_dim
        assert torch.equal(cloned_sculptor.structure, sculptor.structure)
        assert torch.equal(cloned_sculptor.colors, sculptor.colors)
        
        # Modifying clone shouldn't affect original
        cloned_sculptor.reset()
        assert not torch.equal(cloned_sculptor.structure, sculptor.structure)


class TestPyTorchSculptorEquivalence:
    """Test equivalence with original NumPy implementation."""
    
    def test_generation_equivalence_structure(self):
        """Test that PyTorch and NumPy implementations produce similar structures."""
        # Set same random seed for both
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create PyTorch sculptor
        pytorch_sculptor = PyTorchSculptor(
            void_dim=16,
            edges=(1, 0.3, 0.5),
            planes=(1, 0.3, 0.5),
            pipes=(0, 0.3, 0.5),  # Skip pipes for simpler comparison
            grid=(1, 4),
            device="cpu",
        )
        
        # Create NumPy sculptor with same parameters
        numpy_sculptor = Sculptor(
            void_dim=16,
            edges=(1, 0.3, 0.5),
            planes=(1, 0.3, 0.5),
            pipes=(0, 0.3, 0.5),
            grid=(1, 4),
        )
        
        # Generate sculptures
        pytorch_structure, pytorch_colors = pytorch_sculptor.generate_sculpture()
        numpy_structure, numpy_colors = numpy_sculptor.generate_sculpture()
        
        # Convert PyTorch tensors to numpy for comparison
        pytorch_structure_np = pytorch_structure.cpu().numpy()
        pytorch_colors_np = pytorch_colors.cpu().numpy()
        
        # Check that both have similar fill percentages (within reasonable range)
        pytorch_fill = np.sum(pytorch_structure_np > 0) / pytorch_structure_np.size
        numpy_fill = np.sum(numpy_structure > 0) / numpy_structure.size
        
        # Allow for some variation due to randomness
        assert abs(pytorch_fill - numpy_fill) < 0.2  # Within 20%
        
        # Check that both have non-zero structures
        assert pytorch_fill > 0
        assert numpy_fill > 0
    
    def test_memory_efficiency(self):
        """Test memory efficiency of PyTorch implementation."""
        sculptor = PyTorchSculptor(void_dim=32, sparse_mode=True)
        
        # Generate sparse sculpture
        sculptor.generate_sculpture()
        
        # Check memory usage
        memory_usage = sculptor.get_memory_usage()
        assert memory_usage["allocated"] < 1.0  # Should be less than 1GB for 32^3 volume
        
        # Test memory optimization
        sculptor.optimize_memory()
        optimized_memory = sculptor.get_memory_usage()
        assert optimized_memory["allocated"] <= memory_usage["allocated"]


class TestPyTorchSculptorFactory:
    """Test factory function."""
    
    def test_factory_function(self):
        """Test the factory function."""
        sculptor = create_pytorch_sculptor(void_dim=16, device="cpu")
        
        assert isinstance(sculptor, PyTorchSculptor)
        assert sculptor.void_dim == 16
        assert sculptor.device == "cpu"


# Performance benchmarks (optional, run with pytest -m benchmark)
@pytest.mark.benchmark
class TestPyTorchSculptorPerformance:
    """Performance benchmarks for PyTorchSculptor."""
    
    def test_generation_speed_cpu(self, benchmark):
        """Benchmark sculpture generation on CPU."""
        sculptor = PyTorchSculptor(void_dim=32, device="cpu")
        
        def generate():
            sculptor.reset()
            return sculptor.generate_sculpture()
        
        result = benchmark(generate)
        assert result is not None
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_generation_speed_gpu(self, benchmark):
        """Benchmark sculpture generation on GPU."""
        sculptor = PyTorchSculptor(void_dim=32, device="cuda")
        
        def generate():
            sculptor.reset()
            return sculptor.generate_sculpture()
        
        result = benchmark(generate)
        assert result is not None
    
    def test_sparse_conversion_speed(self, benchmark):
        """Benchmark sparse tensor conversion."""
        sculptor = PyTorchSculptor(void_dim=32)
        sculptor.generate_sculpture()
        
        def convert_sparse():
            sculptor.to_dense()
            sculptor.to_sparse()
            return sculptor
        
        result = benchmark(convert_sparse)
        assert result is not None


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])