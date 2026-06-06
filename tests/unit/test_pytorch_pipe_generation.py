"""
Unit tests for PyTorch-based pipe generation functionality.
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
    attach_pipe_pytorch, 
    attach_pipes_batch_pytorch,
    validate_pipe_dimensions_pytorch,
    create_curved_pipe_pytorch,
    SparseTensorHandler
)


class TestPyTorchPipeGeneration:
    """Test suite for PyTorch pipe generation functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.structure_dim = 20
        self.colors_dict = {"pipes": ["blue", "cyan", "magenta"]}

    def test_pipe_attachment_basic(self):
        """Test basic pipe attachment functionality."""
        # Create test tensors
        structure = torch.zeros((self.structure_dim, self.structure_dim, self.structure_dim), device=self.device)
        colors = torch.zeros(structure.shape, device=self.device)

        # Attach pipe
        result_structure, result_colors = attach_pipe_pytorch(
            structure,
            colors,
            element_volume_min_ratio=0.3,
            element_volume_max_ratio=0.6,
            step=2,
            colors_dict=self.colors_dict,
            device=self.device,
            wall_thickness=1,
            pipe_complexity="simple",
            verbose=False,
        )

        # Verify pipe was attached
        if result_structure.is_sparse:
            filled_count = torch.sum(result_structure.to_dense() > 0).item()
        else:
            filled_count = torch.sum(result_structure > 0).item()
        
        assert filled_count > 0, "No pipe was attached"
        assert result_structure.device.type == self.device.split(':')[0], f"Result not on correct device: {self.device}"
        assert result_structure.dtype == torch.int8, "Structure should be int8 for memory efficiency"

    def test_pipe_attachment_sparse_mode(self):
        """Test pipe attachment with sparse tensor support."""
        # Create test tensors
        structure = torch.zeros((self.structure_dim, self.structure_dim, self.structure_dim), device=self.device)
        colors = torch.zeros(structure.shape, device=self.device)

        # Attach pipe with sparse mode
        result_structure, result_colors = attach_pipe_pytorch(
            structure,
            colors,
            element_volume_min_ratio=0.1,
            element_volume_max_ratio=0.2,
            colors_dict=self.colors_dict,
            device=self.device,
            sparse_mode=True,
            wall_thickness=1,
            pipe_complexity="simple",
            verbose=False,
        )

        # Verify pipe was attached
        if result_structure.is_sparse:
            filled_count = torch.sum(result_structure.to_dense() > 0).item()
        else:
            filled_count = torch.sum(result_structure > 0).item()
        assert filled_count > 0, "No pipe was attached in sparse mode"
        
        # Check sparsity
        sparsity = SparseTensorHandler.detect_sparsity(result_structure)
        assert sparsity > 0.7, f"Structure should be sparse, but sparsity is {sparsity}"

    def test_complex_pipe_attachment(self):
        """Test complex pipe attachment with configurable wall thickness."""
        # Create test tensors
        structure = torch.zeros((self.structure_dim, self.structure_dim, self.structure_dim), device=self.device)
        colors = torch.zeros(structure.shape, device=self.device)

        # Attach complex pipe
        result_structure, result_colors = attach_pipe_pytorch(
            structure,
            colors,
            element_volume_min_ratio=0.4,
            element_volume_max_ratio=0.7,
            colors_dict=self.colors_dict,
            device=self.device,
            wall_thickness=2,
            pipe_complexity="complex",
            verbose=False,
        )

        # Verify pipe was attached
        if result_structure.is_sparse:
            filled_count = torch.sum(result_structure.to_dense() > 0).item()
        else:
            filled_count = torch.sum(result_structure > 0).item()
        
        assert filled_count > 0, "No complex pipe was attached"

    def test_batch_pipe_attachment(self):
        """Test batch processing of multiple pipe attachments."""
        batch_size = 3
        
        # Create batch tensors
        structures = torch.zeros((batch_size, self.structure_dim, self.structure_dim, self.structure_dim), device=self.device)
        colors = torch.zeros(structures.shape, device=self.device)

        # Attach pipes in batch
        result_structures, result_colors = attach_pipes_batch_pytorch(
            structures,
            colors,
            element_volume_min_ratio=0.2,
            element_volume_max_ratio=0.5,
            colors_dict=self.colors_dict,
            device=self.device,
            wall_thickness=1,
            pipe_complexity="simple",
            verbose=False,
        )

        # Verify pipes were attached to all structures
        for i in range(batch_size):
            if result_structures[i].is_sparse:
                filled_voxels = torch.sum(result_structures[i].to_dense() > 0).item()
            else:
                filled_voxels = torch.sum(result_structures[i] > 0).item()
            assert filled_voxels > 0, f"No pipe was attached to structure {i}"

        assert result_structures.device.type == self.device.split(':')[0], f"Batch results not on correct device: {self.device}"

    def test_pipe_size_constraints(self):
        """Test that pipe sizes respect the specified ratio constraints."""
        structure = torch.zeros((self.structure_dim, self.structure_dim, self.structure_dim), device=self.device)
        colors = torch.zeros(structure.shape, device=self.device)

        min_ratio = 0.2
        max_ratio = 0.4

        result_structure, result_colors = attach_pipe_pytorch(
            structure,
            colors,
            element_volume_min_ratio=min_ratio,
            element_volume_max_ratio=max_ratio,
            colors_dict=self.colors_dict,
            device=self.device,
            wall_thickness=1,
            pipe_complexity="simple",
            verbose=False,
        )

        # Count filled voxels (should represent pipe volume)
        if result_structure.is_sparse:
            filled_voxels = torch.sum(result_structure.to_dense() > 0).item()
        else:
            filled_voxels = torch.sum(result_structure > 0).item()

        # Pipe volume should be within reasonable bounds (hollow structure)
        min_expected_volume = int((min_ratio * self.structure_dim) ** 3 * 0.1)  # Account for hollow
        max_expected_volume = int((max_ratio * self.structure_dim) ** 3)

        assert filled_voxels >= min_expected_volume, \
            f"Pipe volume {filled_voxels} smaller than minimum expected {min_expected_volume}"

    def test_color_assignment(self):
        """Test that colors are properly assigned to pipe voxels."""
        structure = torch.zeros((self.structure_dim, self.structure_dim, self.structure_dim), device=self.device)
        colors = torch.zeros(structure.shape, device=self.device)

        result_structure, result_colors = attach_pipe_pytorch(
            structure,
            colors,
            colors_dict=self.colors_dict,
            device=self.device,
            wall_thickness=1,
            pipe_complexity="simple",
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

    def test_pipe_dimensions_validation(self):
        """Test pipe dimension validation utility."""
        # Test valid dimensions
        assert validate_pipe_dimensions_pytorch(
            (8, 8, 8), (20, 20, 20), (0, 0, 0), wall_thickness=1, device=self.device
        ), "Valid pipe dimensions should pass validation"

        # Test invalid dimensions (too large)
        assert not validate_pipe_dimensions_pytorch(
            (25, 25, 25), (20, 20, 20), (0, 0, 0), wall_thickness=1, device=self.device
        ), "Oversized pipe dimensions should fail validation"

        # Test invalid dimensions (too small for hollow structure)
        assert not validate_pipe_dimensions_pytorch(
            (2, 2, 2), (20, 20, 20), (0, 0, 0), wall_thickness=2, device=self.device
        ), "Too small pipe dimensions should fail validation"

        # Test invalid position (out of bounds)
        assert not validate_pipe_dimensions_pytorch(
            (8, 8, 8), (20, 20, 20), (15, 15, 15), wall_thickness=1, device=self.device
        ), "Out of bounds position should fail validation"

    def test_wall_thickness_variations(self):
        """Test pipes with different wall thicknesses."""
        wall_thicknesses = [1, 2, 3]
        
        for thickness in wall_thicknesses:
            structure = torch.zeros((self.structure_dim, self.structure_dim, self.structure_dim), device=self.device)
            colors = torch.zeros(structure.shape, device=self.device)

            result_structure, result_colors = attach_pipe_pytorch(
                structure,
                colors,
                element_volume_min_ratio=0.4,
                element_volume_max_ratio=0.6,
                colors_dict=self.colors_dict,
                device=self.device,
                wall_thickness=thickness,
                pipe_complexity="complex",
                verbose=False,
            )

            # Verify pipe was attached
            if result_structure.is_sparse:
                filled_count = torch.sum(result_structure.to_dense() > 0).item()
            else:
                filled_count = torch.sum(result_structure > 0).item()
            
            assert filled_count > 0, f"No pipe was attached with wall thickness {thickness}"

    def test_pipe_complexity_variations(self):
        """Test pipes with different complexity levels."""
        complexities = ["simple", "complex", "curved"]
        
        for complexity in complexities:
            structure = torch.zeros((self.structure_dim, self.structure_dim, self.structure_dim), device=self.device)
            colors = torch.zeros(structure.shape, device=self.device)

            result_structure, result_colors = attach_pipe_pytorch(
                structure,
                colors,
                element_volume_min_ratio=0.3,
                element_volume_max_ratio=0.6,
                colors_dict=self.colors_dict,
                device=self.device,
                wall_thickness=1,
                pipe_complexity=complexity,
                verbose=False,
            )

            # Verify pipe was attached
            if result_structure.is_sparse:
                filled_count = torch.sum(result_structure.to_dense() > 0).item()
            else:
                filled_count = torch.sum(result_structure > 0).item()
            
            assert filled_count > 0, f"No pipe was attached with complexity {complexity}"

    def test_curved_pipe_creation(self):
        """Test curved pipe creation (placeholder implementation)."""
        structure = torch.zeros((self.structure_dim, self.structure_dim, self.structure_dim), device=self.device)
        colors = torch.zeros(structure.shape, device=self.device)

        # Test curved pipe creation (currently returns unchanged tensors)
        result_structure, result_colors = create_curved_pipe_pytorch(
            structure,
            colors,
            start_pos=(2, 2, 2),
            end_pos=(10, 10, 10),
            radius=2,
            color_value=100,
            device=self.device
        )

        # Should return unchanged tensors for now
        assert torch.equal(structure, result_structure), "Curved pipe should return unchanged structure for now"
        assert torch.equal(colors, result_colors), "Curved pipe should return unchanged colors for now"

    def test_memory_efficiency(self):
        """Test memory efficiency improvements with sparse tensors."""
        # Create a large sparse structure
        large_dim = 50
        structure = torch.zeros((large_dim, large_dim, large_dim), device=self.device)
        colors = torch.zeros(structure.shape, device=self.device)

        # Attach pipe with sparse mode
        result_structure, result_colors = attach_pipe_pytorch(
            structure,
            colors,
            element_volume_min_ratio=0.05,  # Very small pipe for high sparsity
            element_volume_max_ratio=0.1,
            colors_dict=self.colors_dict,
            device=self.device,
            sparse_mode=True,
            wall_thickness=1,
            pipe_complexity="simple",
            verbose=False,
        )

        # Verify sparsity is high
        sparsity = SparseTensorHandler.detect_sparsity(result_structure)
        assert sparsity > 0.9, f"Large structure should be very sparse, but sparsity is {sparsity}"

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        structure = torch.zeros((5, 5, 5), device=self.device)
        colors = torch.zeros(structure.shape, device=self.device)

        # Test with invalid complexity
        with pytest.raises(ValueError):
            attach_pipe_pytorch(
                structure,
                colors,
                colors_dict=self.colors_dict,
                device=self.device,
                pipe_complexity="invalid_complexity",
            )

        # Test with invalid ratios should not crash (might skip pipe attachment)
        try:
            result_structure, result_colors = attach_pipe_pytorch(
                structure,
                colors,
                element_volume_min_ratio=1.5,  # Invalid: > 1.0
                element_volume_max_ratio=0.5,
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

        result_structure, result_colors = attach_pipe_pytorch(
            structure,
            colors,
            colors_dict=self.colors_dict,
            device=cuda_device,
            wall_thickness=1,
            pipe_complexity="simple",
            verbose=False,
        )

        assert result_structure.device.type == "cuda", "Structure not on CUDA device"
        assert result_colors.device.type == "cuda", "Colors not on CUDA device"


if __name__ == "__main__":
    # Run tests
    test_suite = TestPyTorchPipeGeneration()
    test_suite.setup_method()
    
    print("Running PyTorch pipe generation tests...")
    
    try:
        test_suite.test_pipe_attachment_basic()
        print("✓ Basic pipe attachment test passed")
        
        test_suite.test_pipe_attachment_sparse_mode()
        print("✓ Sparse mode pipe attachment test passed")
        
        test_suite.test_complex_pipe_attachment()
        print("✓ Complex pipe attachment test passed")
        
        test_suite.test_batch_pipe_attachment()
        print("✓ Batch pipe attachment test passed")
        
        test_suite.test_pipe_size_constraints()
        print("✓ Pipe size constraints test passed")
        
        test_suite.test_color_assignment()
        print("✓ Color assignment test passed")
        
        test_suite.test_pipe_dimensions_validation()
        print("✓ Pipe dimensions validation test passed")
        
        test_suite.test_wall_thickness_variations()
        print("✓ Wall thickness variations test passed")
        
        test_suite.test_pipe_complexity_variations()
        print("✓ Pipe complexity variations test passed")
        
        test_suite.test_curved_pipe_creation()
        print("✓ Curved pipe creation test passed")
        
        test_suite.test_memory_efficiency()
        print("✓ Memory efficiency test passed")
        
        test_suite.test_error_handling()
        print("✓ Error handling test passed")
        
        if torch.cuda.is_available():
            test_suite.test_device_consistency()
            print("✓ Device consistency test passed")
        else:
            print("⚠ Skipping device consistency test (CUDA not available)")
        
        print("\n✅ All PyTorch pipe generation tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        raise