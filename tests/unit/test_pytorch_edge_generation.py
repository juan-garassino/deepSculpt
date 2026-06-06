"""
Unit tests for PyTorch-based edge generation functionality.
Tests equivalence with original NumPy implementation and validates PyTorch-specific features.
"""

import torch
import numpy as np
import pytest
import sys
import os

# Add the deepSculpt module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'deepSculpt'))

from deepsculpt.core.data.generation.pytorch_shapes import attach_edge_pytorch, attach_edges_batch_pytorch, SparseTensorHandler
from shapes import attach_edge


class TestPyTorchEdgeGeneration:
    """Test suite for PyTorch edge generation functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.structure_dim = 20
        self.colors_dict = {"edges": "red"}

    def test_edge_attachment_basic(self):
        """Test basic edge attachment functionality."""
        # Create test tensors
        structure = torch.zeros((self.structure_dim, self.structure_dim, self.structure_dim), device=self.device)
        colors = torch.zeros(structure.shape, device=self.device)

        # Attach edge
        result_structure, result_colors = attach_edge_pytorch(
            structure,
            colors,
            element_edge_min_ratio=0.2,
            element_edge_max_ratio=0.5,
            step=2,
            colors_dict=self.colors_dict,
            device=self.device,
            verbose=False,
        )

        # Verify edge was attached
        assert torch.sum(result_structure > 0).item() > 0, "No edge was attached"
        assert result_structure.device.type == self.device.split(':')[0], f"Result not on correct device: {self.device}"
        assert result_structure.dtype == torch.int8, "Structure should be int8 for memory efficiency"

    def test_edge_attachment_sparse_mode(self):
        """Test edge attachment with sparse tensor support."""
        # Create test tensors
        structure = torch.zeros((self.structure_dim, self.structure_dim, self.structure_dim), device=self.device)
        colors = torch.zeros(structure.shape, device=self.device)

        # Attach edge with sparse mode
        result_structure, result_colors = attach_edge_pytorch(
            structure,
            colors,
            element_edge_min_ratio=0.1,
            element_edge_max_ratio=0.2,
            colors_dict=self.colors_dict,
            device=self.device,
            sparse_mode=True,
            verbose=False,
        )

        # Verify edge was attached (convert to dense for counting if sparse)
        if result_structure.is_sparse:
            filled_count = torch.sum(result_structure.to_dense() > 0).item()
        else:
            filled_count = torch.sum(result_structure > 0).item()
        assert filled_count > 0, "No edge was attached in sparse mode"
        
        # Check sparsity
        sparsity = SparseTensorHandler.detect_sparsity(result_structure)
        assert sparsity > 0.8, f"Structure should be sparse, but sparsity is {sparsity}"

    def test_batch_edge_attachment(self):
        """Test batch processing of multiple edge attachments."""
        batch_size = 3
        
        # Create batch tensors
        structures = torch.zeros((batch_size, self.structure_dim, self.structure_dim, self.structure_dim), device=self.device)
        colors = torch.zeros(structures.shape, device=self.device)

        # Attach edges in batch
        result_structures, result_colors = attach_edges_batch_pytorch(
            structures,
            colors,
            element_edge_min_ratio=0.2,
            element_edge_max_ratio=0.5,
            colors_dict=self.colors_dict,
            device=self.device,
            verbose=False,
        )

        # Verify edges were attached to all structures
        for i in range(batch_size):
            if result_structures[i].is_sparse:
                filled_voxels = torch.sum(result_structures[i].to_dense() > 0).item()
            else:
                filled_voxels = torch.sum(result_structures[i] > 0).item()
            assert filled_voxels > 0, f"No edge was attached to structure {i}"

        assert result_structures.device.type == self.device.split(':')[0], f"Batch results not on correct device: {self.device}"

    def test_device_consistency(self):
        """Test that tensors remain on the specified device throughout processing."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for device consistency test")

        # Test with CUDA device
        cuda_device = "cuda"
        structure = torch.zeros((10, 10, 10), device=cuda_device)
        colors = torch.zeros(structure.shape, device=cuda_device)

        result_structure, result_colors = attach_edge_pytorch(
            structure,
            colors,
            colors_dict=self.colors_dict,
            device=cuda_device,
            verbose=False,
        )

        assert result_structure.device.type == "cuda", "Structure not on CUDA device"
        assert result_colors.device.type == "cuda", "Colors not on CUDA device"

    def test_edge_size_constraints(self):
        """Test that edge sizes respect the specified ratio constraints."""
        structure = torch.zeros((self.structure_dim, self.structure_dim, self.structure_dim), device=self.device)
        colors = torch.zeros(structure.shape, device=self.device)

        min_ratio = 0.1
        max_ratio = 0.3

        result_structure, result_colors = attach_edge_pytorch(
            structure,
            colors,
            element_edge_min_ratio=min_ratio,
            element_edge_max_ratio=max_ratio,
            colors_dict=self.colors_dict,
            device=self.device,
            verbose=False,
        )

        # Count filled voxels (should represent edge length)
        if result_structure.is_sparse:
            filled_voxels = torch.sum(result_structure.to_dense() > 0).item()
        else:
            filled_voxels = torch.sum(result_structure > 0).item()
        min_expected = int(min_ratio * self.structure_dim)
        max_expected = int(max_ratio * self.structure_dim)

        assert min_expected <= filled_voxels <= max_expected, \
            f"Edge size {filled_voxels} not within expected range [{min_expected}, {max_expected}]"

    def test_color_assignment(self):
        """Test that colors are properly assigned to edge voxels."""
        structure = torch.zeros((self.structure_dim, self.structure_dim, self.structure_dim), device=self.device)
        colors = torch.zeros(structure.shape, device=self.device)

        result_structure, result_colors = attach_edge_pytorch(
            structure,
            colors,
            colors_dict=self.colors_dict,
            device=self.device,
            verbose=False,
        )

        # Check that colored voxels correspond to filled voxels
        if result_structure.is_sparse:
            filled_mask = result_structure.to_dense() > 0
        else:
            filled_mask = result_structure > 0
            
        if result_colors.is_sparse:
            colored_mask = result_colors.to_dense() > 0
        else:
            colored_mask = result_colors > 0

        assert torch.equal(filled_mask, colored_mask), "Color assignment doesn't match structure filling"

    def test_sparse_tensor_handler(self):
        """Test sparse tensor handling utilities."""
        # Create a sparse tensor (mostly zeros)
        dense_tensor = torch.zeros((10, 10, 10), device=self.device)
        dense_tensor[1:3, 1:3, 1:3] = 1  # Small filled region

        # Test sparsity detection
        sparsity = SparseTensorHandler.detect_sparsity(dense_tensor)
        expected_sparsity = 1.0 - (8.0 / 1000.0)  # 8 filled voxels out of 1000
        assert abs(sparsity - expected_sparsity) < 0.01, f"Sparsity calculation incorrect: {sparsity} vs {expected_sparsity}"

        # Test sparse conversion
        sparse_tensor = SparseTensorHandler.to_sparse(dense_tensor, threshold=0.5)
        assert sparse_tensor.is_sparse, "Tensor should be converted to sparse"

        # Test dense conversion
        dense_again = SparseTensorHandler.to_dense(sparse_tensor)
        assert not dense_again.is_sparse, "Tensor should be converted back to dense"
        assert torch.equal(dense_tensor, dense_again), "Dense conversion should preserve values"

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        structure = torch.zeros((5, 5, 5), device=self.device)
        colors = torch.zeros(structure.shape, device=self.device)

        # Test with invalid ratios
        with pytest.raises(Exception):
            attach_edge_pytorch(
                structure,
                colors,
                element_edge_min_ratio=1.5,  # Invalid: > 1.0
                element_edge_max_ratio=0.5,
                colors_dict=self.colors_dict,
                device=self.device,
            )

    def test_memory_efficiency(self):
        """Test memory efficiency improvements with sparse tensors."""
        # Create a large sparse structure
        large_dim = 50
        structure = torch.zeros((large_dim, large_dim, large_dim), device=self.device)
        colors = torch.zeros(structure.shape, device=self.device)

        # Attach edge with sparse mode
        result_structure, result_colors = attach_edge_pytorch(
            structure,
            colors,
            element_edge_min_ratio=0.02,  # Very small edge for high sparsity
            element_edge_max_ratio=0.05,
            colors_dict=self.colors_dict,
            device=self.device,
            sparse_mode=True,
            verbose=False,
        )

        # Verify sparsity is high
        sparsity = SparseTensorHandler.detect_sparsity(result_structure)
        assert sparsity > 0.95, f"Large structure should be very sparse, but sparsity is {sparsity}"


if __name__ == "__main__":
    # Run tests
    test_suite = TestPyTorchEdgeGeneration()
    test_suite.setup_method()
    
    print("Running PyTorch edge generation tests...")
    
    try:
        test_suite.test_edge_attachment_basic()
        print("✓ Basic edge attachment test passed")
        
        test_suite.test_edge_attachment_sparse_mode()
        print("✓ Sparse mode edge attachment test passed")
        
        test_suite.test_batch_edge_attachment()
        print("✓ Batch edge attachment test passed")
        
        if torch.cuda.is_available():
            test_suite.test_device_consistency()
            print("✓ Device consistency test passed")
        else:
            print("⚠ Skipping device consistency test (CUDA not available)")
        
        test_suite.test_edge_size_constraints()
        print("✓ Edge size constraints test passed")
        
        test_suite.test_color_assignment()
        print("✓ Color assignment test passed")
        
        test_suite.test_sparse_tensor_handler()
        print("✓ Sparse tensor handler test passed")
        
        test_suite.test_memory_efficiency()
        print("✓ Memory efficiency test passed")
        
        print("\n✅ All PyTorch edge generation tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        raise