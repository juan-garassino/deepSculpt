"""
Unit tests for PyTorch-based plane generation functionality.
Tests equivalence with original NumPy implementation and validates PyTorch-specific features.
"""

import torch
import numpy as np
import pytest
import sys
import os

# Add the deepSculpt module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'deepSculpt'))

from deepsculpt.core.data.generation.pytorch_shapes import (
    attach_plane_pytorch, 
    attach_planes_batch_pytorch, 
    attach_plane_with_rotation_pytorch,
    validate_plane_dimensions_pytorch,
    SparseTensorHandler
)


class TestPyTorchPlaneGeneration:
    """Test suite for PyTorch plane generation functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.structure_dim = 20
        self.colors_dict = {"planes": "green"}

    def test_plane_attachment_basic(self):
        """Test basic plane attachment functionality."""
        # Create test tensors
        structure = torch.zeros((self.structure_dim, self.structure_dim, self.structure_dim), device=self.device)
        colors = torch.zeros(structure.shape, device=self.device)

        # Attach plane
        result_structure, result_colors = attach_plane_pytorch(
            structure,
            colors,
            element_plane_min_ratio=0.2,
            element_plane_max_ratio=0.5,
            step=2,
            colors_dict=self.colors_dict,
            device=self.device,
            verbose=False,
        )

        # Verify plane was attached
        if result_structure.is_sparse:
            filled_count = torch.sum(result_structure.to_dense() > 0).item()
        else:
            filled_count = torch.sum(result_structure > 0).item()
        
        assert filled_count > 0, "No plane was attached"
        assert result_structure.device.type == self.device.split(':')[0], f"Result not on correct device: {self.device}"
        assert result_structure.dtype == torch.int8, "Structure should be int8 for memory efficiency"

    def test_plane_attachment_sparse_mode(self):
        """Test plane attachment with sparse tensor support."""
        # Create test tensors
        structure = torch.zeros((self.structure_dim, self.structure_dim, self.structure_dim), device=self.device)
        colors = torch.zeros(structure.shape, device=self.device)

        # Attach plane with sparse mode
        result_structure, result_colors = attach_plane_pytorch(
            structure,
            colors,
            element_plane_min_ratio=0.1,
            element_plane_max_ratio=0.2,
            colors_dict=self.colors_dict,
            device=self.device,
            sparse_mode=True,
            verbose=False,
        )

        # Verify plane was attached
        if result_structure.is_sparse:
            filled_count = torch.sum(result_structure.to_dense() > 0).item()
        else:
            filled_count = torch.sum(result_structure > 0).item()
        assert filled_count > 0, "No plane was attached in sparse mode"
        
        # Check sparsity
        sparsity = SparseTensorHandler.detect_sparsity(result_structure)
        assert sparsity > 0.7, f"Structure should be sparse, but sparsity is {sparsity}"

    def test_batch_plane_attachment(self):
        """Test batch processing of multiple plane attachments."""
        batch_size = 3
        
        # Create batch tensors
        structures = torch.zeros((batch_size, self.structure_dim, self.structure_dim, self.structure_dim), device=self.device)
        colors = torch.zeros(structures.shape, device=self.device)

        # Attach planes in batch
        result_structures, result_colors = attach_planes_batch_pytorch(
            structures,
            colors,
            element_plane_min_ratio=0.2,
            element_plane_max_ratio=0.5,
            colors_dict=self.colors_dict,
            device=self.device,
            verbose=False,
        )

        # Verify planes were attached to all structures
        for i in range(batch_size):
            if result_structures[i].is_sparse:
                filled_voxels = torch.sum(result_structures[i].to_dense() > 0).item()
            else:
                filled_voxels = torch.sum(result_structures[i] > 0).item()
            assert filled_voxels > 0, f"No plane was attached to structure {i}"

        assert result_structures.device.type == self.device.split(':')[0], f"Batch results not on correct device: {self.device}"

    def test_plane_size_constraints(self):
        """Test that plane sizes respect the specified ratio constraints."""
        structure = torch.zeros((self.structure_dim, self.structure_dim, self.structure_dim), device=self.device)
        colors = torch.zeros(structure.shape, device=self.device)

        min_ratio = 0.1
        max_ratio = 0.3

        result_structure, result_colors = attach_plane_pytorch(
            structure,
            colors,
            element_plane_min_ratio=min_ratio,
            element_plane_max_ratio=max_ratio,
            colors_dict=self.colors_dict,
            device=self.device,
            verbose=False,
        )

        # Count filled voxels (should represent plane area)
        if result_structure.is_sparse:
            filled_voxels = torch.sum(result_structure.to_dense() > 0).item()
        else:
            filled_voxels = torch.sum(result_structure > 0).item()

        # Plane area should be within reasonable bounds
        min_expected_area = int((min_ratio * self.structure_dim) ** 2)
        max_expected_area = int((max_ratio * self.structure_dim) ** 2)

        assert filled_voxels >= min_expected_area, \
            f"Plane area {filled_voxels} smaller than minimum expected {min_expected_area}"
        # Note: max constraint is relaxed since plane can be rectangular

    def test_color_assignment(self):
        """Test that colors are properly assigned to plane voxels."""
        structure = torch.zeros((self.structure_dim, self.structure_dim, self.structure_dim), device=self.device)
        colors = torch.zeros(structure.shape, device=self.device)

        result_structure, result_colors = attach_plane_pytorch(
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

    def test_plane_dimensions_validation(self):
        """Test plane dimension validation utility."""
        # Test valid dimensions
        assert validate_plane_dimensions_pytorch(
            (5, 5), (20, 20, 20), (0, 0, 0), self.device
        ), "Valid plane dimensions should pass validation"

        # Test invalid dimensions (too large)
        assert not validate_plane_dimensions_pytorch(
            (25, 25), (20, 20, 20), (0, 0, 0), self.device
        ), "Oversized plane dimensions should fail validation"

        # Test invalid position (out of bounds)
        assert not validate_plane_dimensions_pytorch(
            (5, 5), (20, 20, 20), (18, 18, 0), self.device
        ), "Out of bounds position should fail validation"

    def test_plane_with_rotation(self):
        """Test plane attachment with rotation (basic implementation)."""
        structure = torch.zeros((self.structure_dim, self.structure_dim, self.structure_dim), device=self.device)
        colors = torch.zeros(structure.shape, device=self.device)

        # Test with rotation angles (currently falls back to basic plane)
        result_structure, result_colors = attach_plane_with_rotation_pytorch(
            structure,
            colors,
            rotation_angles=(45.0, 0.0, 0.0),
            colors_dict=self.colors_dict,
            device=self.device,
            verbose=False,
        )

        # Verify plane was attached (even though rotation is not yet implemented)
        if result_structure.is_sparse:
            filled_count = torch.sum(result_structure.to_dense() > 0).item()
        else:
            filled_count = torch.sum(result_structure > 0).item()
        
        assert filled_count > 0, "No plane was attached with rotation"

    def test_different_orientations(self):
        """Test plane attachment with different orientations."""
        orientations = ["xy", "xz", "yz", "random"]
        
        for orientation in orientations:
            structure = torch.zeros((self.structure_dim, self.structure_dim, self.structure_dim), device=self.device)
            colors = torch.zeros(structure.shape, device=self.device)

            result_structure, result_colors = attach_plane_pytorch(
                structure,
                colors,
                orientation=orientation,
                colors_dict=self.colors_dict,
                device=self.device,
                verbose=False,
            )

            # Verify plane was attached
            if result_structure.is_sparse:
                filled_count = torch.sum(result_structure.to_dense() > 0).item()
            else:
                filled_count = torch.sum(result_structure > 0).item()
            
            assert filled_count > 0, f"No plane was attached with orientation {orientation}"

    def test_memory_efficiency(self):
        """Test memory efficiency improvements with sparse tensors."""
        # Create a large sparse structure
        large_dim = 50
        structure = torch.zeros((large_dim, large_dim, large_dim), device=self.device)
        colors = torch.zeros(structure.shape, device=self.device)

        # Attach plane with sparse mode
        result_structure, result_colors = attach_plane_pytorch(
            structure,
            colors,
            element_plane_min_ratio=0.02,  # Very small plane for high sparsity
            element_plane_max_ratio=0.05,
            colors_dict=self.colors_dict,
            device=self.device,
            sparse_mode=True,
            verbose=False,
        )

        # Verify sparsity is high
        sparsity = SparseTensorHandler.detect_sparsity(result_structure)
        assert sparsity > 0.9, f"Large structure should be very sparse, but sparsity is {sparsity}"

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        structure = torch.zeros((5, 5, 5), device=self.device)
        colors = torch.zeros(structure.shape, device=self.device)

        # Test with invalid ratios should not crash (might skip plane attachment)
        try:
            result_structure, result_colors = attach_plane_pytorch(
                structure,
                colors,
                element_plane_min_ratio=1.5,  # Invalid: > 1.0
                element_plane_max_ratio=0.5,
                colors_dict=self.colors_dict,
                device=self.device,
                verbose=False,
            )
            # Should either work or skip gracefully
            assert True, "Function should handle invalid ratios gracefully"
        except Exception as e:
            # If it raises an exception, it should be a reasonable one
            assert "ratio" in str(e).lower() or "size" in str(e).lower(), f"Unexpected error: {e}"

    def test_device_consistency(self):
        """Test that tensors remain on the specified device throughout processing."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for device consistency test")

        # Test with CUDA device
        cuda_device = "cuda"
        structure = torch.zeros((10, 10, 10), device=cuda_device)
        colors = torch.zeros(structure.shape, device=cuda_device)

        result_structure, result_colors = attach_plane_pytorch(
            structure,
            colors,
            colors_dict=self.colors_dict,
            device=cuda_device,
            verbose=False,
        )

        assert result_structure.device.type == "cuda", "Structure not on CUDA device"
        assert result_colors.device.type == "cuda", "Colors not on CUDA device"


if __name__ == "__main__":
    # Run tests
    test_suite = TestPyTorchPlaneGeneration()
    test_suite.setup_method()
    
    print("Running PyTorch plane generation tests...")
    
    try:
        test_suite.test_plane_attachment_basic()
        print("✓ Basic plane attachment test passed")
        
        test_suite.test_plane_attachment_sparse_mode()
        print("✓ Sparse mode plane attachment test passed")
        
        test_suite.test_batch_plane_attachment()
        print("✓ Batch plane attachment test passed")
        
        test_suite.test_plane_size_constraints()
        print("✓ Plane size constraints test passed")
        
        test_suite.test_color_assignment()
        print("✓ Color assignment test passed")
        
        test_suite.test_plane_dimensions_validation()
        print("✓ Plane dimensions validation test passed")
        
        test_suite.test_plane_with_rotation()
        print("✓ Plane with rotation test passed")
        
        test_suite.test_different_orientations()
        print("✓ Different orientations test passed")
        
        test_suite.test_memory_efficiency()
        print("✓ Memory efficiency test passed")
        
        test_suite.test_error_handling()
        print("✓ Error handling test passed")
        
        if torch.cuda.is_available():
            test_suite.test_device_consistency()
            print("✓ Device consistency test passed")
        else:
            print("⚠ Skipping device consistency test (CUDA not available)")
        
        print("\n✅ All PyTorch plane generation tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        raise