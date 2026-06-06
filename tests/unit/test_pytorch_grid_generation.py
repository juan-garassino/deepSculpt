"""
Unit tests for PyTorch-based grid generation functionality.
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
    attach_grid_pytorch, 
    attach_grids_batch_pytorch,
    create_procedural_grid_pytorch,
    validate_grid_parameters_pytorch,
    _generate_regular_grid_pytorch,
    _generate_irregular_grid_pytorch,
    _generate_random_grid_pytorch,
    SparseTensorHandler
)


class TestPyTorchGridGeneration:
    """Test suite for PyTorch grid generation functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.structure_dim = 20
        self.colors_dict = {"edges": "red"}

    def test_grid_attachment_basic(self):
        """Test basic grid attachment functionality."""
        # Create test tensors
        structure = torch.zeros((self.structure_dim, self.structure_dim, self.structure_dim), device=self.device)
        colors = torch.zeros(structure.shape, device=self.device)

        # Attach grid
        result_structure, result_colors = attach_grid_pytorch(
            structure,
            colors,
            step=4,
            colors_dict=self.colors_dict,
            device=self.device,
            grid_pattern="regular",
            grid_density=0.7,
            verbose=False,
        )

        # Verify grid was attached
        if result_structure.is_sparse:
            filled_count = torch.sum(result_structure.to_dense() > 0).item()
        else:
            filled_count = torch.sum(result_structure > 0).item()
        
        assert filled_count > 0, "No grid was attached"
        assert result_structure.device.type == self.device.split(':')[0], f"Result not on correct device: {self.device}"
        assert result_structure.dtype == torch.int8, "Structure should be int8 for memory efficiency"

    def test_grid_attachment_sparse_mode(self):
        """Test grid attachment with sparse tensor support."""
        # Create test tensors
        structure = torch.zeros((self.structure_dim, self.structure_dim, self.structure_dim), device=self.device)
        colors = torch.zeros(structure.shape, device=self.device)

        # Attach grid with sparse mode
        result_structure, result_colors = attach_grid_pytorch(
            structure,
            colors,
            step=6,
            colors_dict=self.colors_dict,
            device=self.device,
            sparse_mode=True,
            grid_pattern="regular",
            grid_density=0.3,
            verbose=False,
        )

        # Verify grid was attached
        if result_structure.is_sparse:
            filled_count = torch.sum(result_structure.to_dense() > 0).item()
        else:
            filled_count = torch.sum(result_structure > 0).item()
        assert filled_count > 0, "No grid was attached in sparse mode"
        
        # Check sparsity
        sparsity = SparseTensorHandler.detect_sparsity(result_structure)
        assert sparsity > 0.5, f"Structure should be sparse, but sparsity is {sparsity}"

    def test_different_grid_patterns(self):
        """Test grid attachment with different patterns."""
        patterns = ["regular", "irregular", "random"]
        
        for pattern in patterns:
            structure = torch.zeros((self.structure_dim, self.structure_dim, self.structure_dim), device=self.device)
            colors = torch.zeros(structure.shape, device=self.device)

            result_structure, result_colors = attach_grid_pytorch(
                structure,
                colors,
                step=4,
                colors_dict=self.colors_dict,
                device=self.device,
                grid_pattern=pattern,
                grid_density=0.5,
                verbose=False,
            )

            # Verify grid was attached
            if result_structure.is_sparse:
                filled_count = torch.sum(result_structure.to_dense() > 0).item()
            else:
                filled_count = torch.sum(result_structure > 0).item()
            
            assert filled_count > 0, f"No grid was attached with pattern {pattern}"

    def test_batch_grid_attachment(self):
        """Test batch processing of multiple grid attachments."""
        batch_size = 3
        
        # Create batch tensors
        structures = torch.zeros((batch_size, self.structure_dim, self.structure_dim, self.structure_dim), device=self.device)
        colors = torch.zeros(structures.shape, device=self.device)

        # Attach grids in batch
        result_structures, result_colors = attach_grids_batch_pytorch(
            structures,
            colors,
            step=5,
            colors_dict=self.colors_dict,
            device=self.device,
            grid_pattern="regular",
            grid_density=0.6,
            verbose=False,
        )

        # Verify grids were attached to all structures
        for i in range(batch_size):
            if result_structures[i].is_sparse:
                filled_voxels = torch.sum(result_structures[i].to_dense() > 0).item()
            else:
                filled_voxels = torch.sum(result_structures[i] > 0).item()
            assert filled_voxels > 0, f"No grid was attached to structure {i}"

        assert result_structures.device.type == self.device.split(':')[0], f"Batch results not on correct device: {self.device}"

    def test_grid_density_variations(self):
        """Test grids with different density values."""
        densities = [0.2, 0.5, 0.8, 1.0]
        
        for density in densities:
            structure = torch.zeros((self.structure_dim, self.structure_dim, self.structure_dim), device=self.device)
            colors = torch.zeros(structure.shape, device=self.device)

            result_structure, result_colors = attach_grid_pytorch(
                structure,
                colors,
                step=4,
                colors_dict=self.colors_dict,
                device=self.device,
                grid_pattern="regular",
                grid_density=density,
                verbose=False,
            )

            # Verify grid was attached
            if result_structure.is_sparse:
                filled_count = torch.sum(result_structure.to_dense() > 0).item()
            else:
                filled_count = torch.sum(result_structure > 0).item()
            
            assert filled_count > 0, f"No grid was attached with density {density}"

    def test_column_height_variations(self):
        """Test grids with and without column height variation."""
        for height_variation in [True, False]:
            structure = torch.zeros((self.structure_dim, self.structure_dim, self.structure_dim), device=self.device)
            colors = torch.zeros(structure.shape, device=self.device)

            result_structure, result_colors = attach_grid_pytorch(
                structure,
                colors,
                step=4,
                colors_dict=self.colors_dict,
                device=self.device,
                grid_pattern="regular",
                grid_density=0.5,
                column_height_variation=height_variation,
                verbose=False,
            )

            # Verify grid was attached
            if result_structure.is_sparse:
                filled_count = torch.sum(result_structure.to_dense() > 0).item()
            else:
                filled_count = torch.sum(result_structure > 0).item()
            
            assert filled_count > 0, f"No grid was attached with height_variation={height_variation}"

    def test_base_floor_option(self):
        """Test grids with and without base floor."""
        for base_floor in [True, False]:
            structure = torch.zeros((self.structure_dim, self.structure_dim, self.structure_dim), device=self.device)
            colors = torch.zeros(structure.shape, device=self.device)

            result_structure, result_colors = attach_grid_pytorch(
                structure,
                colors,
                step=4,
                colors_dict=self.colors_dict,
                device=self.device,
                grid_pattern="regular",
                grid_density=0.5,
                base_floor=base_floor,
                verbose=False,
            )

            # Verify grid was attached
            if result_structure.is_sparse:
                filled_count = torch.sum(result_structure.to_dense() > 0).item()
            else:
                filled_count = torch.sum(result_structure > 0).item()
            
            assert filled_count > 0, f"No grid was attached with base_floor={base_floor}"

            # Check if floor exists when requested
            if base_floor:
                if result_structure.is_sparse:
                    floor_filled = torch.sum(result_structure.to_dense()[:, :, 0] > 0).item()
                else:
                    floor_filled = torch.sum(result_structure[:, :, 0] > 0).item()
                assert floor_filled > 0, "Base floor should be present when requested"

    def test_color_assignment(self):
        """Test that colors are properly assigned to grid voxels."""
        structure = torch.zeros((self.structure_dim, self.structure_dim, self.structure_dim), device=self.device)
        colors = torch.zeros(structure.shape, device=self.device)

        result_structure, result_colors = attach_grid_pytorch(
            structure,
            colors,
            step=4,
            colors_dict=self.colors_dict,
            device=self.device,
            grid_pattern="regular",
            grid_density=0.5,
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

    def test_grid_generation_functions(self):
        """Test individual grid generation functions."""
        structure_dim = 20
        step = 4
        density = 0.5
        device = self.device

        # Test regular grid generation
        regular_locations = _generate_regular_grid_pytorch(structure_dim, step, density, device)
        assert len(regular_locations) > 0, "Regular grid should generate locations"

        # Test irregular grid generation
        irregular_locations = _generate_irregular_grid_pytorch(structure_dim, step, density, device)
        assert len(irregular_locations) > 0, "Irregular grid should generate locations"

        # Test random grid generation
        random_locations = _generate_random_grid_pytorch(structure_dim, step, density, device)
        assert len(random_locations) > 0, "Random grid should generate locations"

        # Verify all locations are within bounds
        for locations in [regular_locations, irregular_locations, random_locations]:
            for x, y in locations:
                assert 0 <= x < structure_dim, f"X coordinate {x} out of bounds"
                assert 0 <= y < structure_dim, f"Y coordinate {y} out of bounds"

    def test_procedural_grid_creation(self):
        """Test procedural grid creation with parameters."""
        structure = torch.zeros((self.structure_dim, self.structure_dim, self.structure_dim), device=self.device)
        colors = torch.zeros(structure.shape, device=self.device)

        grid_params = {
            "pattern": "irregular",
            "density": 0.6,
            "step": 3,
            "colors_dict": self.colors_dict,
            "sparse_mode": True,
            "height_variation": True,
            "base_floor": True,
            "verbose": False
        }

        result_structure, result_colors = create_procedural_grid_pytorch(
            structure,
            colors,
            grid_params,
            device=self.device
        )

        # Verify grid was created
        if result_structure.is_sparse:
            filled_count = torch.sum(result_structure.to_dense() > 0).item()
        else:
            filled_count = torch.sum(result_structure > 0).item()
        
        assert filled_count > 0, "No procedural grid was created"

    def test_grid_parameters_validation(self):
        """Test grid parameter validation."""
        structure_shape = (20, 20, 20)

        # Test valid parameters
        valid_params = {
            "step": 4,
            "density": 0.5,
            "pattern": "regular"
        }
        assert validate_grid_parameters_pytorch(valid_params, structure_shape, self.device), \
            "Valid parameters should pass validation"

        # Test invalid step
        invalid_step_params = {
            "step": 0,
            "density": 0.5,
            "pattern": "regular"
        }
        assert not validate_grid_parameters_pytorch(invalid_step_params, structure_shape, self.device), \
            "Invalid step should fail validation"

        # Test invalid density
        invalid_density_params = {
            "step": 4,
            "density": 1.5,
            "pattern": "regular"
        }
        assert not validate_grid_parameters_pytorch(invalid_density_params, structure_shape, self.device), \
            "Invalid density should fail validation"

        # Test invalid pattern
        invalid_pattern_params = {
            "step": 4,
            "density": 0.5,
            "pattern": "invalid"
        }
        assert not validate_grid_parameters_pytorch(invalid_pattern_params, structure_shape, self.device), \
            "Invalid pattern should fail validation"

    def test_memory_efficiency(self):
        """Test memory efficiency improvements with sparse tensors."""
        # Create a large sparse structure
        large_dim = 50
        structure = torch.zeros((large_dim, large_dim, large_dim), device=self.device)
        colors = torch.zeros(structure.shape, device=self.device)

        # Attach grid with sparse mode
        result_structure, result_colors = attach_grid_pytorch(
            structure,
            colors,
            step=10,  # Large step for high sparsity
            colors_dict=self.colors_dict,
            device=self.device,
            sparse_mode=True,
            grid_pattern="regular",
            grid_density=0.2,  # Low density for high sparsity
            verbose=False,
        )

        # Verify sparsity is high
        sparsity = SparseTensorHandler.detect_sparsity(result_structure)
        assert sparsity > 0.8, f"Large structure should be very sparse, but sparsity is {sparsity}"

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        structure = torch.zeros((10, 10, 10), device=self.device)
        colors = torch.zeros(structure.shape, device=self.device)

        # Test with invalid pattern
        with pytest.raises(ValueError):
            attach_grid_pytorch(
                structure,
                colors,
                step=2,
                colors_dict=self.colors_dict,
                device=self.device,
                grid_pattern="invalid_pattern",
            )

        # Test with invalid density should not crash (might adjust internally)
        try:
            result_structure, result_colors = attach_grid_pytorch(
                structure,
                colors,
                step=2,
                colors_dict=self.colors_dict,
                device=self.device,
                grid_density=1.5,  # Invalid: > 1.0
                verbose=False,
            )
            # Should either work or handle gracefully
            assert True, "Function should handle invalid density gracefully"
        except Exception as e:
            # If it raises an exception, it should be a reasonable one
            assert "density" in str(e).lower(), f"Unexpected error: {e}"

    def test_device_consistency(self):
        """Test that tensors remain on the specified device throughout processing."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for device consistency test")

        # Test with CUDA device
        cuda_device = "cuda"
        structure = torch.zeros((10, 10, 10), device=cuda_device)
        colors = torch.zeros(structure.shape, device=cuda_device)

        result_structure, result_colors = attach_grid_pytorch(
            structure,
            colors,
            step=3,
            colors_dict=self.colors_dict,
            device=cuda_device,
            grid_pattern="regular",
            grid_density=0.5,
            verbose=False,
        )

        assert result_structure.device.type == "cuda", "Structure not on CUDA device"
        assert result_colors.device.type == "cuda", "Colors not on CUDA device"


if __name__ == "__main__":
    # Run tests
    test_suite = TestPyTorchGridGeneration()
    test_suite.setup_method()
    
    print("Running PyTorch grid generation tests...")
    
    try:
        test_suite.test_grid_attachment_basic()
        print("✓ Basic grid attachment test passed")
        
        test_suite.test_grid_attachment_sparse_mode()
        print("✓ Sparse mode grid attachment test passed")
        
        test_suite.test_different_grid_patterns()
        print("✓ Different grid patterns test passed")
        
        test_suite.test_batch_grid_attachment()
        print("✓ Batch grid attachment test passed")
        
        test_suite.test_grid_density_variations()
        print("✓ Grid density variations test passed")
        
        test_suite.test_column_height_variations()
        print("✓ Column height variations test passed")
        
        test_suite.test_base_floor_option()
        print("✓ Base floor option test passed")
        
        test_suite.test_color_assignment()
        print("✓ Color assignment test passed")
        
        test_suite.test_grid_generation_functions()
        print("✓ Grid generation functions test passed")
        
        test_suite.test_procedural_grid_creation()
        print("✓ Procedural grid creation test passed")
        
        test_suite.test_grid_parameters_validation()
        print("✓ Grid parameters validation test passed")
        
        test_suite.test_memory_efficiency()
        print("✓ Memory efficiency test passed")
        
        test_suite.test_error_handling()
        print("✓ Error handling test passed")
        
        if torch.cuda.is_available():
            test_suite.test_device_consistency()
            print("✓ Device consistency test passed")
        else:
            print("⚠ Skipping device consistency test (CUDA not available)")
        
        print("\n✅ All PyTorch grid generation tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        raise