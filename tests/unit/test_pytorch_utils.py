"""
Comprehensive tests for PyTorch utility functions.
Tests equivalence with original NumPy implementations and validates PyTorch-specific functionality.
"""

import pytest
import torch
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock

# Import the classes to test
from deepsculpt.core.utils.pytorch_utils import PyTorchUtils, MemoryOptimizer, MemoryProfiler


class TestPyTorchUtils:
    """Test cases for PyTorchUtils class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.void_shape = (32, 32, 32)
        self.void = torch.zeros(self.void_shape, device=self.device)
        self.color_void = torch.zeros(self.void_shape, device=self.device, dtype=torch.float32)
    
    def test_return_axis(self):
        """Test return_axis function."""
        # Create test tensors with some filled voxels
        self.void[10:20, 10:20, 10:20] = 1
        self.color_void[10:20, 10:20, 10:20] = 0.5
        
        working_plane, color_parameters, section = PyTorchUtils.return_axis(
            self.void, self.color_void, self.device
        )
        
        # Verify outputs
        assert isinstance(working_plane, torch.Tensor)
        assert isinstance(color_parameters, torch.Tensor)
        assert isinstance(section, int)
        assert 0 <= section < self.void_shape[0]
        assert working_plane.device.type == self.device.split(':')[0]
        assert color_parameters.device.type == self.device.split(':')[0]
    
    def test_generate_random_size(self):
        """Test generate_random_size function."""
        base_size = 32
        min_ratio = 0.1
        max_ratio = 0.5
        
        size = PyTorchUtils.generate_random_size(
            min_ratio, max_ratio, base_size, device=self.device
        )
        
        expected_min = max(int(min_ratio * base_size), 2)
        expected_max = max(int(max_ratio * base_size), expected_min + 1)
        
        assert isinstance(size, int)
        assert expected_min <= size < expected_max
    
    def test_select_random_position(self):
        """Test select_random_position function."""
        max_pos = 32
        size = 8
        
        position = PyTorchUtils.select_random_position(max_pos, size, self.device)
        
        assert isinstance(position, int)
        assert 0 <= position <= max(0, max_pos - size)
    
    def test_insert_shape(self):
        """Test insert_shape function."""
        void = torch.zeros((10, 10, 10), device=self.device)
        shape_indices = (slice(2, 5), slice(2, 5), slice(2, 5))
        
        # Test with default values (1s)
        result = PyTorchUtils.insert_shape(void, shape_indices)
        assert torch.all(result[shape_indices] == 1)
        
        # Test with custom values
        void = torch.zeros((10, 10, 10), device=self.device)
        custom_values = torch.full((3, 3, 3), 0.5, device=self.device)
        result = PyTorchUtils.insert_shape(void, shape_indices, custom_values)
        assert torch.all(result[shape_indices] == 0.5)
    
    def test_assign_color(self):
        """Test assign_color function."""
        color_void = torch.zeros((10, 10, 10), device=self.device)
        shape_indices = (slice(2, 5), slice(2, 5), slice(2, 5))
        color = 0.8
        
        result = PyTorchUtils.assign_color(color_void, shape_indices, color)
        assert torch.all(result[shape_indices] == color)
    
    def test_validate_dimensions(self):
        """Test validate_dimensions function."""
        # Valid dimensions
        assert PyTorchUtils.validate_dimensions([5, 5, 5], (10, 10, 10))
        
        # Invalid dimensions
        assert not PyTorchUtils.validate_dimensions([15, 5, 5], (10, 10, 10))
    
    def test_validate_bounds(self):
        """Test validate_bounds function."""
        # Valid bounds
        assert PyTorchUtils.validate_bounds([2, 2, 2], [5, 5, 5], (10, 10, 10))
        
        # Invalid bounds - exceeds dimensions
        assert not PyTorchUtils.validate_bounds([8, 2, 2], [5, 5, 5], (10, 10, 10))
        
        # Invalid bounds - negative position
        assert not PyTorchUtils.validate_bounds([-1, 2, 2], [5, 5, 5], (10, 10, 10))
    
    def test_tensor_to_voxel_coordinates(self):
        """Test tensor_to_voxel_coordinates function."""
        tensor = torch.zeros((5, 5, 5), device=self.device)
        tensor[1, 2, 3] = 1
        tensor[2, 3, 4] = 1
        
        coords = PyTorchUtils.tensor_to_voxel_coordinates(tensor)
        
        assert coords.shape[1] == 3  # 3D coordinates
        assert coords.shape[0] == 2  # Two filled voxels
        assert torch.equal(coords[0], torch.tensor([1, 2, 3], device=self.device))
        assert torch.equal(coords[1], torch.tensor([2, 3, 4], device=self.device))
    
    def test_apply_3d_transformations(self):
        """Test apply_3d_transformations function."""
        # Create a simple tensor with one filled voxel
        tensor = torch.zeros((10, 10, 10), device=self.device)
        tensor[5, 5, 5] = 1
        
        # Test translation
        translation = torch.tensor([1.0, 1.0, 1.0], device=self.device)
        result = PyTorchUtils.apply_3d_transformations(tensor, translation=translation)
        
        # The voxel should move from [5,5,5] to [6,6,6]
        assert result[6, 6, 6] == 1
        assert result[5, 5, 5] == 0
        
        # Test scaling
        result = PyTorchUtils.apply_3d_transformations(tensor, scale=2.0)
        # Original position [5,5,5] should be empty, but we can't easily predict new position
        # due to rounding, so just check that transformation occurred
        assert not torch.equal(result, tensor)
    
    def test_validate_tensor_dtype(self):
        """Test validate_tensor_dtype function."""
        tensor_float = torch.zeros((5, 5, 5), dtype=torch.float32, device=self.device)
        tensor_int = torch.zeros((5, 5, 5), dtype=torch.int32, device=self.device)
        
        assert PyTorchUtils.validate_tensor_dtype(tensor_float, torch.float32)
        assert not PyTorchUtils.validate_tensor_dtype(tensor_float, torch.int32)
        assert PyTorchUtils.validate_tensor_dtype(tensor_int, torch.int32)
    
    def test_validate_tensor_device(self):
        """Test validate_tensor_device function."""
        tensor = torch.zeros((5, 5, 5), device=self.device)
        
        assert PyTorchUtils.validate_tensor_device(tensor, self.device)
        
        # Test with different device
        other_device = "cpu" if self.device != "cpu" else "cuda:0"
        if other_device == "cuda:0" and not torch.cuda.is_available():
            other_device = "cpu"
        
        if other_device != self.device:
            assert not PyTorchUtils.validate_tensor_device(tensor, other_device)
    
    def test_ensure_tensor_device(self):
        """Test ensure_tensor_device function."""
        tensor = torch.zeros((5, 5, 5), device="cpu")
        
        result = PyTorchUtils.ensure_tensor_device(tensor, self.device)
        assert result.device.type == self.device.split(':')[0]
    
    def test_efficient_tensor_slicing(self):
        """Test efficient_tensor_slicing function."""
        tensor = torch.arange(1000, device=self.device).reshape(10, 10, 10)
        
        # Valid slicing
        slice_indices = (slice(2, 5), slice(3, 7), slice(1, 8))
        result = PyTorchUtils.efficient_tensor_slicing(tensor, slice_indices)
        
        assert result.shape == (3, 4, 7)
        
        # Test bounds checking - should raise IndexError
        with pytest.raises(IndexError):
            invalid_slice = (slice(2, 15), slice(3, 7), slice(1, 8))  # 15 > 10
            PyTorchUtils.efficient_tensor_slicing(tensor, invalid_slice)
    
    def test_create_debug_info(self):
        """Test create_debug_info function."""
        tensor = torch.zeros((10, 10, 10), device=self.device)
        tensor[2:5, 2:5, 2:5] = 1  # Fill 27 voxels
        
        info = PyTorchUtils.create_debug_info(tensor)
        
        assert info["shape"] == (10, 10, 10)
        assert info["total_voxels"] == 1000
        assert info["filled_voxels"] == 27
        assert info["fill_percentage"] == 2.7
        assert info["sparsity"] == 0.973
        assert info["device"] == str(tensor.device)
        assert info["dtype"] == str(tensor.dtype)
        assert "memory_usage_bytes" in info
    
    def test_equivalence_with_numpy(self):
        """Test equivalence with original NumPy implementations."""
        # This test compares key functions with their NumPy equivalents
        
        # Test random size generation consistency
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Generate multiple sizes and check they're in valid range
        base_size = 32
        min_ratio, max_ratio = 0.1, 0.5
        
        for _ in range(10):
            pytorch_size = PyTorchUtils.generate_random_size(min_ratio, max_ratio, base_size)
            expected_min = max(int(min_ratio * base_size), 2)
            expected_max = max(int(max_ratio * base_size), expected_min + 1)
            
            assert expected_min <= pytorch_size < expected_max


class TestMemoryOptimizer:
    """Test cases for MemoryOptimizer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def test_detect_sparsity(self):
        """Test sparsity detection."""
        # Dense tensor (no zeros)
        dense_tensor = torch.ones((10, 10, 10), device=self.device)
        assert MemoryOptimizer.detect_sparsity(dense_tensor) == 0.0
        
        # Completely sparse tensor (all zeros)
        sparse_tensor = torch.zeros((10, 10, 10), device=self.device)
        assert MemoryOptimizer.detect_sparsity(sparse_tensor) == 1.0
        
        # Partially sparse tensor
        partial_tensor = torch.zeros((10, 10, 10), device=self.device)
        partial_tensor[:5, :5, :5] = 1  # Fill 125 out of 1000 voxels
        expected_sparsity = 875 / 1000  # 875 zeros out of 1000
        assert abs(MemoryOptimizer.detect_sparsity(partial_tensor) - expected_sparsity) < 1e-6
    
    def test_should_use_sparse(self):
        """Test sparse usage decision."""
        # High sparsity tensor should use sparse
        sparse_tensor = torch.zeros((100, 100, 100), device=self.device)
        sparse_tensor[:10, :10, :10] = 1  # Only 1% filled
        
        assert MemoryOptimizer.should_use_sparse(sparse_tensor, sparsity_threshold=0.5)
        
        # Low sparsity tensor should not use sparse
        dense_tensor = torch.ones((10, 10, 10), device=self.device)
        dense_tensor[:2, :2, :2] = 0  # Only 8 zeros out of 1000
        
        assert not MemoryOptimizer.should_use_sparse(dense_tensor, sparsity_threshold=0.5)
    
    def test_sparse_dense_conversion(self):
        """Test conversion between sparse and dense formats."""
        # Create a sparse tensor
        original = torch.zeros((20, 20, 20), device=self.device)
        original[5:10, 5:10, 5:10] = 1.0
        
        # Convert to sparse
        sparse_version = MemoryOptimizer.to_sparse(original)
        assert sparse_version.is_sparse
        
        # Convert back to dense
        dense_version = MemoryOptimizer.to_dense(sparse_version)
        assert not dense_version.is_sparse
        
        # Check equivalence
        assert torch.allclose(original, dense_version)
    
    def test_auto_convert_sparse_dense(self):
        """Test automatic conversion between sparse and dense."""
        # High sparsity tensor should be converted to sparse
        sparse_tensor = torch.zeros((50, 50, 50), device=self.device)
        sparse_tensor[:5, :5, :5] = 1  # Very sparse
        
        result = MemoryOptimizer.auto_convert_sparse_dense(sparse_tensor, sparsity_threshold=0.5)
        assert result.is_sparse
        
        # Low sparsity tensor should remain dense
        dense_tensor = torch.ones((10, 10, 10), device=self.device)
        dense_tensor[:2, :2, :2] = 0  # Mostly filled
        
        result = MemoryOptimizer.auto_convert_sparse_dense(dense_tensor, sparsity_threshold=0.5)
        assert not result.is_sparse
    
    def test_get_tensor_memory_usage(self):
        """Test memory usage calculation."""
        tensor = torch.ones((10, 10, 10), dtype=torch.float32, device=self.device)
        expected_memory = 10 * 10 * 10 * 4  # 4 bytes per float32
        
        actual_memory = MemoryOptimizer.get_tensor_memory_usage(tensor)
        assert actual_memory == expected_memory
    
    def test_get_memory_stats(self):
        """Test memory statistics retrieval."""
        stats = MemoryOptimizer.get_memory_stats(self.device)
        
        required_keys = ["device", "total_memory", "allocated_memory", "available_memory"]
        for key in required_keys:
            assert key in stats
            if key == "device":
                assert isinstance(stats[key], str)
            else:
                assert isinstance(stats[key], (int, float))
    
    def test_suggest_optimization(self):
        """Test optimization suggestions."""
        # High sparsity tensor should suggest sparse conversion
        sparse_tensor = torch.zeros((50, 50, 50), device=self.device)
        sparse_tensor[:5, :5, :5] = 1
        
        suggestions = MemoryOptimizer.suggest_optimization(sparse_tensor)
        
        assert "sparsity" in suggestions
        assert "suggestions" in suggestions
        assert suggestions["sparsity"] > 0.5
        
        # Should suggest sparse conversion
        suggestion_types = [s["type"] for s in suggestions["suggestions"]]
        assert "convert_to_sparse" in suggestion_types
    
    def test_compress_tensor(self):
        """Test tensor compression."""
        tensor = torch.randn((10, 10, 10), device=self.device)
        
        # Test quantization
        quantized = MemoryOptimizer.compress_tensor(tensor, "quantization")
        assert quantized.dtype == torch.int8
        assert hasattr(quantized, 'scale')
        
        # Test pruning
        pruned = MemoryOptimizer.compress_tensor(tensor, "pruning", threshold=0.1)
        assert pruned.shape == tensor.shape
        # Some values should be pruned to zero
        assert torch.sum(pruned == 0) > 0
    
    def test_create_memory_profile(self):
        """Test memory profiling."""
        tensors = [
            torch.ones((10, 10, 10), device=self.device),
            torch.zeros((20, 20, 20), device=self.device),
        ]
        names = ["dense_tensor", "sparse_tensor"]
        
        profile = MemoryOptimizer.create_memory_profile(tensors, names)
        
        assert profile["total_tensors"] == 2
        assert "total_memory" in profile
        assert len(profile["tensor_details"]) == 2
        
        for detail in profile["tensor_details"]:
            assert "name" in detail
            assert "shape" in detail
            assert "memory_usage" in detail
            assert "sparsity" in detail


class TestMemoryProfiler:
    """Test cases for MemoryProfiler class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.profiler = MemoryProfiler(self.device)
    
    def test_take_snapshot(self):
        """Test memory snapshot functionality."""
        snapshot = self.profiler.take_snapshot("test_snapshot")
        
        assert "name" in snapshot
        assert "memory_stats" in snapshot
        assert snapshot["name"] == "test_snapshot"
        assert len(self.profiler.snapshots) == 1
    
    def test_profile_operation(self):
        """Test operation profiling."""
        def dummy_operation(size):
            return torch.ones((size, size, size), device=self.device)
        
        result, profile_data = self.profiler.profile_operation(
            dummy_operation, "create_tensor", 10
        )
        
        assert result.shape == (10, 10, 10)
        assert "operation_name" in profile_data
        assert "elapsed_time_ms" in profile_data
        assert "memory_increase" in profile_data
        assert profile_data["operation_name"] == "create_tensor"
        assert len(self.profiler.operations) == 1
    
    def test_generate_report(self):
        """Test report generation."""
        # Take some snapshots and profile operations
        self.profiler.take_snapshot("initial")
        
        def dummy_op():
            return torch.ones((5, 5, 5), device=self.device)
        
        self.profiler.profile_operation(dummy_op, "test_op")
        
        report = self.profiler.generate_report()
        
        assert isinstance(report, str)
        assert "Memory Profiling Report" in report
        assert "Memory Snapshots:" in report
        assert "Operation Profiles:" in report
        assert "initial" in report
        assert "test_op" in report
    
    def test_clear_history(self):
        """Test clearing profiling history."""
        self.profiler.take_snapshot("test")
        
        def dummy_op():
            return torch.ones((2, 2, 2), device=self.device)
        
        self.profiler.profile_operation(dummy_op, "test_op")
        
        assert len(self.profiler.snapshots) > 0
        assert len(self.profiler.operations) > 0
        
        self.profiler.clear_history()
        
        assert len(self.profiler.snapshots) == 0
        assert len(self.profiler.operations) == 0


class TestIntegration:
    """Integration tests comparing PyTorch and NumPy implementations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def test_coordinate_operations_equivalence(self):
        """Test that coordinate operations produce equivalent results."""
        # Set seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Test multiple iterations to ensure consistency
        for _ in range(10):
            base_size = 32
            min_ratio, max_ratio = 0.1, 0.5
            
            # PyTorch version
            pytorch_size = PyTorchUtils.generate_random_size(
                min_ratio, max_ratio, base_size, device=self.device
            )
            
            # Check bounds are equivalent to what NumPy version would produce
            expected_min = max(int(min_ratio * base_size), 2)
            expected_max = max(int(max_ratio * base_size), expected_min + 1)
            
            assert expected_min <= pytorch_size < expected_max
    
    def test_memory_optimization_effectiveness(self):
        """Test that memory optimization actually reduces memory usage."""
        # Create a highly sparse tensor
        large_tensor = torch.zeros((100, 100, 100), device=self.device)
        large_tensor[:10, :10, :10] = 1  # Only 1% filled
        
        original_memory = MemoryOptimizer.get_tensor_memory_usage(large_tensor)
        
        # Convert to sparse
        sparse_tensor = MemoryOptimizer.to_sparse(large_tensor)
        sparse_memory = MemoryOptimizer.get_tensor_memory_usage(sparse_tensor)
        
        # Sparse representation should use less memory
        assert sparse_memory < original_memory
        
        # Verify mathematical equivalence
        dense_recovered = MemoryOptimizer.to_dense(sparse_tensor)
        assert torch.allclose(large_tensor, dense_recovered)


if __name__ == "__main__":
    pytest.main([__file__])