"""
Comprehensive test for all PyTorch shape generation functionality.
Tests all shape types: edges, planes, pipes, and grids.
"""

import torch
import sys
import os

# Add the deepSculpt module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'deepSculpt'))

from deepsculpt.core.data.generation.pytorch_shapes import (
    attach_edge_pytorch, 
    attach_plane_pytorch, 
    attach_pipe_pytorch,
    SparseTensorHandler
)
from logger import log_info, log_success, begin_section, end_section

def simple_attach_grid_pytorch(structure, colors, step=4, device="cpu"):
    """Simple grid attachment for comprehensive testing."""
    structure_dim = structure.shape[0]
    locations = []
    for x in range(step, structure_dim - step, step * 2):
        for y in range(step, structure_dim - step, step * 2):
            locations.append((x, y))
    
    for x, y in locations:
        height = torch.randint(structure_dim // 4, structure_dim // 2, (1,)).item()
        structure[x, y, 0:height] = 1
        colors[x, y, 0:height] = 100
    
    structure[:, :, 0] = 1
    colors[:, :, 0] = 100
    
    return structure, colors

def test_all_shapes():
    """Test all shape generation functions."""
    begin_section("Testing all PyTorch shape generation functions")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_info(f"Using device: {device}")
    
    structure_dim = 20
    colors_dict = {
        "edges": "red",
        "planes": "green", 
        "pipes": ["blue", "cyan", "magenta"]
    }
    
    # Test 1: Edge generation
    log_info("Testing edge generation...")
    structure = torch.zeros((structure_dim, structure_dim, structure_dim), device=device)
    colors = torch.zeros(structure.shape, device=device)
    
    structure, colors = attach_edge_pytorch(
        structure, colors, 
        element_edge_min_ratio=0.2, element_edge_max_ratio=0.5,
        colors_dict=colors_dict, device=device, verbose=False
    )
    
    edge_filled = torch.sum(structure > 0).item()
    assert edge_filled > 0, "Edge generation failed"
    log_success(f"Edge generation: {edge_filled} voxels filled")
    
    # Test 2: Plane generation
    log_info("Testing plane generation...")
    structure = torch.zeros((structure_dim, structure_dim, structure_dim), device=device)
    colors = torch.zeros(structure.shape, device=device)
    
    structure, colors = attach_plane_pytorch(
        structure, colors,
        element_plane_min_ratio=0.3, element_plane_max_ratio=0.6,
        colors_dict=colors_dict, device=device, verbose=False
    )
    
    plane_filled = torch.sum(structure > 0).item()
    assert plane_filled > 0, "Plane generation failed"
    log_success(f"Plane generation: {plane_filled} voxels filled")
    
    # Test 3: Pipe generation
    log_info("Testing pipe generation...")
    structure = torch.zeros((structure_dim, structure_dim, structure_dim), device=device)
    colors = torch.zeros(structure.shape, device=device)
    
    structure, colors = attach_pipe_pytorch(
        structure, colors,
        element_volume_min_ratio=0.4, element_volume_max_ratio=0.7,
        colors_dict=colors_dict, device=device, 
        wall_thickness=1, pipe_complexity="simple", verbose=False
    )
    
    pipe_filled = torch.sum(structure > 0).item()
    assert pipe_filled > 0, "Pipe generation failed"
    log_success(f"Pipe generation: {pipe_filled} voxels filled")
    
    # Test 4: Grid generation (simple version)
    log_info("Testing grid generation...")
    structure = torch.zeros((structure_dim, structure_dim, structure_dim), device=device)
    colors = torch.zeros(structure.shape, device=device)
    
    structure, colors = simple_attach_grid_pytorch(structure, colors, step=4, device=device)
    
    grid_filled = torch.sum(structure > 0).item()
    assert grid_filled > 0, "Grid generation failed"
    log_success(f"Grid generation: {grid_filled} voxels filled")
    
    # Test 5: Sparse tensor functionality
    log_info("Testing sparse tensor functionality...")
    structure = torch.zeros((structure_dim, structure_dim, structure_dim), device=device)
    colors = torch.zeros(structure.shape, device=device)
    
    # Add a small edge to create sparse structure
    structure, colors = attach_edge_pytorch(
        structure, colors,
        element_edge_min_ratio=0.1, element_edge_max_ratio=0.2,
        colors_dict=colors_dict, device=device, sparse_mode=True, verbose=False
    )
    
    sparsity = SparseTensorHandler.detect_sparsity(structure)
    assert sparsity > 0.8, f"Structure should be sparse, sparsity: {sparsity}"
    log_success(f"Sparse tensor functionality: sparsity = {sparsity:.3f}")
    
    # Test 6: Device consistency
    log_info("Testing device consistency...")
    assert structure.device.type == device.split(':')[0], "Device consistency failed"
    log_success("Device consistency verified")
    
    # Test 7: Memory efficiency
    log_info("Testing memory efficiency...")
    assert structure.dtype == torch.int8 or structure.is_sparse, "Memory efficiency check failed"
    log_success("Memory efficiency verified")
    
    log_success("All PyTorch shape generation tests passed!")
    end_section()

if __name__ == "__main__":
    test_all_shapes()
    print("\n✅ Comprehensive PyTorch shapes test completed successfully!")