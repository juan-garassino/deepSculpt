"""
PyTorch-based Sculpture Generation System for DeepSculpt
This module provides the PyTorchSculptor class that creates complete 3D voxel-based
sculptures using PyTorch tensors instead of NumPy arrays, with support for GPU acceleration,
sparse tensors, and memory optimization.

Key features:
- Complete sculpture generation using PyTorch tensors
- GPU acceleration and automatic device management
- Sparse/dense tensor mode switching for memory optimization
- Shape composition using PyTorch operations
- Method chaining and fluent interface
- Configuration validation and parameter checking
- Memory usage monitoring and optimization
- Sculpture transformation operations
- Save/load functionality with PyTorch tensors

Dependencies:
- torch: For tensor operations and GPU acceleration
- logger.py: For process tracking and status reporting
- pytorch_shapes.py: For PyTorch-based shape generation
- visualization.py: For displaying generated sculptures

Used by:
- collector.py: For batch generation of sculptures
- training pipelines: For data generation during training

Terminology:
- structure: 3D PyTorch tensor representing the sculpture shape
- colors: 3D PyTorch tensor with color/material information
"""

#!/usr/bin/env python3
"""
PyTorch-based Sculpture Generation System for DeepSculpt

This module provides the `PyTorchSculptor` class that creates complete 3D voxel-based
sculptures using PyTorch tensors instead of NumPy arrays, with support for GPU acceleration,
sparse tensors, and memory optimization.

Key Features:
- Complete sculpture generation using PyTorch tensors
- GPU acceleration and automatic device management
- Sparse/dense tensor mode switching for memory optimization
- Shape composition using PyTorch operations
- Method chaining and fluent interface
- Configuration validation and parameter checking
- Memory usage monitoring and optimization
- Sculpture transformation operations
- Save/load functionality with PyTorch tensors

Dependencies:
- torch: For tensor operations and GPU acceleration
- logger.py: For process tracking and status reporting
- pytorch_shapes.py: For PyTorch-based shape generation
- visualization.py: For displaying generated sculptures

Used by:
- collector.py: For batch generation of sculptures
- training pipelines: For data generation during training

Terminology:
- structure: 3D PyTorch tensor representing the sculpture shape
- colors: 3D PyTorch tensor with color/material information
"""

import os
import time
import copy
import random
import torch
from enum import Enum
from typing import Tuple, List, Dict, Any, Optional, Union

from deepsculpt.core.utils.logger import (
    begin_section,
    end_section,
    log_action,
    log_success,
    log_error,
    log_info,
    log_warning,
)

from .pytorch_shapes import (
    ShapeType,
    attach_edge_pytorch,
    attach_plane_pytorch,
    attach_pipe_pytorch,
    attach_grid_pytorch,
    SparseTensorHandler,
    PyTorchUtils,
)


class PyTorchSculptor:
    """
    A PyTorch-based class for creating 3D sculptures with various components.
    Supports GPU acceleration, sparse tensors, and memory optimization.
    """

    def __init__(
        self,
        void_dim: int = 20,
        edges: Tuple[int, float, float] = (1, 0.3, 0.5),
        planes: Tuple[int, float, float] = (1, 0.3, 0.5),
        pipes: Tuple[int, float, float] = (1, 0.3, 0.5),
        grid: Tuple[int, int] = (1, 4),
        colors: Optional[Dict[str, Any]] = None,
        step: int = 1,
        device: str = "auto",
        sparse_mode: bool = False,
        sparse_threshold: float = 0.1,
        memory_limit_gb: float = 4.0,
        verbose: bool = False,
    ):
        """
        Initialize a new PyTorchSculptor instance.

        Args:
            void_dim: The dimension of the structure (cube size)
            edges: Tuple of (count, min_ratio, max_ratio) for edges
            planes: Tuple of (count, min_ratio, max_ratio) for planes
            pipes: Tuple of (count, min_ratio, max_ratio) for pipes
            grid: Tuple of (enable, step) for grid
            colors: Dictionary of colors for different shape types
            step: Step size for shape dimensions
            device: Device to use ("auto", "cuda", "cpu")
            sparse_mode: Whether to use sparse tensors by default
            sparse_threshold: Sparsity threshold for automatic sparse conversion
            memory_limit_gb: Memory limit in GB for automatic optimization
            verbose: Whether to print detailed information
        """
        begin_section("Initializing PyTorchSculptor")

        try:
            # --- Validate parameters ---
            self._validate_init_parameters(
                void_dim, edges, planes, pipes, grid, step,
                sparse_threshold, memory_limit_gb
            )

            # --- Core parameters ---
            self.void_dim = void_dim
            self.edges = edges
            self.planes = planes
            self.pipes = pipes
            self.grid = grid
            self.step = step
            self.sparse_mode = sparse_mode
            self.sparse_threshold = sparse_threshold
            self.memory_limit_gb = memory_limit_gb
            self.verbose = verbose

            # --- Device management ---
            self.device = self._setup_device(device)
            log_info(f"Using device: {self.device}")

            # --- Initialize tensors ---
            self._initialize_tensors()

            # --- Default colors ---
            # Grid columns, floors, and slabs use "edges" color (gray)
            # Structural elements (pipes, edges) stay colorful for contrast
            self.colors_dict = colors or {
                "edges": "red",
                "planes": "yellow",
                "pipes": ["blue", "red"],
                "volumes": ["blue", "red", "yellow"],
            }

            # --- Undo/redo history ---
            self._history: List[Dict[str, Any]] = []
            self._history_index = -1
            self._max_history_size = 10

            # --- Performance tracking ---
            self._memory_usage: Dict[str, float] = {}
            self._generation_stats: Dict[str, Any] = {}

            log_success(
                f"PyTorchSculptor initialized with void_dim={void_dim}, device={self.device}"
            )
            end_section()

        except Exception as e:
            log_error(f"Error initializing PyTorchSculptor: {str(e)}")
            end_section("PyTorchSculptor initialization failed")
            raise

    # ================================
    # --- Validation & Device Setup ---
    # ================================

    def _validate_init_parameters(
        self,
        void_dim: int,
        edges: Tuple[int, float, float],
        planes: Tuple[int, float, float],
        pipes: Tuple[int, float, float],
        grid: Tuple[int, int],
        step: int,
        sparse_threshold: float,
        memory_limit_gb: float,
    ):
        """Validate initialization parameters."""
        if void_dim <= 0:
            raise ValueError("void_dim must be positive")
        if void_dim > 512:
            log_warning(f"Large void_dim ({void_dim}) may cause memory issues")

        for name, params in [("edges", edges), ("planes", planes), ("pipes", pipes)]:
            if len(params) != 3:
                raise ValueError(f"{name} must be a tuple of (count, min_ratio, max_ratio)")
            count, min_ratio, max_ratio = params
            if count < 0:
                raise ValueError(f"{name} count must be non-negative")
            if not (0 <= min_ratio <= max_ratio <= 1):
                raise ValueError(f"{name} ratios must be in [0, 1] with min_ratio <= max_ratio")

        if len(grid) != 2:
            raise ValueError("grid must be a tuple of (enable, step)")
        if grid[0] not in [0, 1]:
            raise ValueError("grid enable must be 0 or 1")
        if grid[1] <= 0:
            raise ValueError("grid step must be positive")

        if step <= 0:
            raise ValueError("step must be positive")
        if not (0 <= sparse_threshold <= 1):
            raise ValueError("sparse_threshold must be in [0, 1]")
        if memory_limit_gb <= 0:
            raise ValueError("memory_limit_gb must be positive")

    def _setup_device(self, device: str) -> str:
        """Setup and validate the compute device."""
        if device == "auto":
            if torch.cuda.is_available():
                log_info("CUDA available, using GPU")
                return "cuda"
            else:
                log_info("CUDA not available, using CPU")
                return "cpu"

        if device == "cuda" and not torch.cuda.is_available():
            log_warning("CUDA requested but not available, falling back to CPU")
            return "cpu"

        return device

    # ================================
    # --- Tensor Initialization ---
    # ================================

    def _initialize_tensors(self):
        """Initialize the structure and colors tensors."""
        # Structure tensor
        self.structure = torch.zeros(
            (self.void_dim, self.void_dim, self.void_dim),
            dtype=torch.int8,
            device=self.device,
        )

        # Colors tensor (int for color indices)
        self.colors = torch.zeros(
            (self.void_dim, self.void_dim, self.void_dim),
            dtype=torch.int16,
            device=self.device,
        )

        # Sparse conversion if requested
        if self.sparse_mode:
            self.structure = SparseTensorHandler.to_sparse(self.structure, self.sparse_threshold)
            self.colors = SparseTensorHandler.to_sparse(self.colors, self.sparse_threshold)

        # Initial memory tracking
        self._update_memory_usage()

    # ================================
    # --- Memory Management ---
    # ================================

    def _update_memory_usage(self):
        """Update memory usage tracking."""
        if self.device == "cuda":
            self._memory_usage = {
                "allocated": torch.cuda.memory_allocated(self.device) / (1024**3),
                "reserved": torch.cuda.memory_reserved(self.device) / (1024**3),
                "max_allocated": torch.cuda.max_memory_allocated(self.device) / (1024**3),
            }
        else:
            # Estimate CPU memory usage
            struct_size = self.structure.numel() * self.structure.element_size() / (1024**3)
            color_size = self.colors.numel() * self.colors.element_size() / (1024**3)
            self._memory_usage = {
                "allocated": struct_size + color_size,
                "reserved": struct_size + color_size,
                "max_allocated": struct_size + color_size,
            }

    def get_memory_usage(self) -> Dict[str, float]:
        """Return current memory usage statistics."""
        self._update_memory_usage()
        return self._memory_usage.copy()

    # ================================
    # --- Sparse/Dense Conversion ---
    # ================================

    def to_sparse(self) -> 'PyTorchSculptor':
        """Convert tensors to sparse representation if sparse enough."""
        begin_section("Converting to sparse tensors")

        try:
            if not self.structure.is_sparse:
                sparsity = SparseTensorHandler.detect_sparsity(self.structure)
                log_info(f"Structure sparsity: {sparsity:.3f}")

                if sparsity > self.sparse_threshold:
                    self.structure = SparseTensorHandler.to_sparse(self.structure, self.sparse_threshold)
                    self.colors = SparseTensorHandler.to_sparse(self.colors, self.sparse_threshold)
                    self.sparse_mode = True
                    log_success("Converted to sparse tensors")
                else:
                    log_info(
                        f"Sparsity ({sparsity:.3f}) below threshold ({self.sparse_threshold}), keeping dense"
                    )
            else:
                log_info("Tensors already sparse")

            self._update_memory_usage()
            end_section()
            return self

        except Exception as e:
            log_error(f"Error converting to sparse: {str(e)}")
            end_section("Sparse conversion failed")
            raise

    def to_dense(self) -> 'PyTorchSculptor':
        """Convert tensors back to dense representation."""
        begin_section("Converting to dense tensors")

        try:
            if self.structure.is_sparse:
                self.structure = SparseTensorHandler.to_dense(self.structure)
                self.colors = SparseTensorHandler.to_dense(self.colors)
                self.sparse_mode = False
                log_success("Converted to dense tensors")
            else:
                log_info("Tensors already dense")

            self._update_memory_usage()
            end_section()
            return self

        except Exception as e:
            log_error(f"Error converting to dense: {str(e)}")
            end_section("Dense conversion failed")
            raise

    def optimize_memory(self) -> 'PyTorchSculptor':
        """Optimize memory usage based on current state and thresholds."""
        begin_section("Optimizing memory usage")

        try:
            current = self.get_memory_usage()
            log_info(f"Current memory usage: {current['allocated']:.2f} GB")

            if current['allocated'] > self.memory_limit_gb:
                log_warning(
                    f"Usage ({current['allocated']:.2f} GB) exceeds limit ({self.memory_limit_gb} GB)"
                )

                if not self.sparse_mode:
                    sparsity = SparseTensorHandler.detect_sparsity(self.structure)
                    if sparsity > self.sparse_threshold:
                        log_action("Converting to sparse tensors to reduce memory usage")
                        self.to_sparse()
                        reduced = self.get_memory_usage()
                        log_success(f"Memory reduced to {reduced['allocated']:.2f} GB")
                    else:
                        log_warning("Cannot optimize further - structure not sparse enough")
                else:
                    log_warning("Already sparse - cannot optimize further")
            else:
                log_info("Memory usage within limits")

            end_section()
            return self

        except Exception as e:
            log_error(f"Error optimizing memory: {str(e)}")
            end_section("Memory optimization failed")
            raise

    def to_device(self, device: str) -> 'PyTorchSculptor':
        """Move tensors to specified device."""
        begin_section(f"Moving tensors to {device}")

        try:
            if device != self.device:
                self.structure = self.structure.to(device)
                self.colors = self.colors.to(device)
                self.device = device
                log_success(f"Moved tensors to {device}")
            else:
                log_info(f"Tensors already on {device}")

            self._update_memory_usage()
            end_section()
            return self

        except Exception as e:
            log_error(f"Error moving to device: {str(e)}")
            end_section("Device transfer failed")
            raise

    # ================================
    # --- Tensor Information & Validation ---
    # ================================

    def get_tensor_info(self) -> Dict[str, Any]:
        """Get detailed information about the current tensors."""
        structure_info = {
            "shape": tuple(self.structure.shape),
            "dtype": str(self.structure.dtype),
            "device": str(self.structure.device),
            "is_sparse": self.structure.is_sparse,
            "memory_mb": self.structure.numel() * self.structure.element_size() / (1024**2),
        }

        colors_info = {
            "shape": tuple(self.colors.shape),
            "dtype": str(self.colors.dtype),
            "device": str(self.colors.device),
            "is_sparse": self.colors.is_sparse,
            "memory_mb": self.colors.numel() * self.colors.element_size() / (1024**2),
        }

        if not self.structure.is_sparse:
            filled = torch.sum(self.structure > 0).item()
            total = self.structure.numel()
            sparsity = 1.0 - (filled / total)
        else:
            sparsity = SparseTensorHandler.detect_sparsity(self.structure)

        return {
            "structure": structure_info,
            "colors": colors_info,
            "sparsity": sparsity,
            "memory_usage": self.get_memory_usage(),
            "sparse_mode": self.sparse_mode,
        }

    def validate_configuration(self) -> bool:
        """Validate current configuration and tensor states."""
        begin_section("Validating configuration")

        try:
            if self.structure.shape != self.colors.shape:
                log_error("Structure and colors tensors have different shapes")
                return False

            if self.structure.device != self.colors.device:
                log_error("Structure and colors tensors on different devices")
                return False

            expected_shape = (self.void_dim, self.void_dim, self.void_dim)
            if self.structure.shape != expected_shape:
                log_error(f"Tensor shape {self.structure.shape} != void_dim {self.void_dim}")
                return False

            usage = self.get_memory_usage()
            if usage['allocated'] > self.memory_limit_gb * 2:
                log_warning(f"Very high memory usage: {usage['allocated']:.2f} GB")

            if self.sparse_mode and not self.structure.is_sparse:
                log_warning("sparse_mode=True but tensors are dense")

            log_success("Configuration validation passed")
            end_section()
            return True

        except Exception as e:
            log_error(f"Error validating configuration: {str(e)}")
            end_section("Configuration validation failed")
            return False

    # ================================
    # --- History (Undo/Redo) ---
    # ================================

    def _save_state_to_history(self):
        """Save current state to history for undo functionality."""
        if len(self._history) >= self._max_history_size:
            self._history.pop(0)

        state = {
            "structure": self.structure.clone(),
            "colors": self.colors.clone(),
            "sparse_mode": self.sparse_mode,
        }

        if self._history_index < len(self._history) - 1:
            self._history = self._history[: self._history_index + 1]

        self._history.append(state)
        self._history_index = len(self._history) - 1

    def __repr__(self) -> str:
        info = self.get_tensor_info()
        return (
            f"PyTorchSculptor(void_dim={self.void_dim}, "
            f"device={self.device}, sparse_mode={self.sparse_mode}, "
            f"sparsity={info['sparsity']:.3f}, "
            f"memory={info['memory_usage']['allocated']:.2f}GB)"
        )

    # ================================
    # --- Sculpture Generation ---
    # ================================

    def generate_sculpture(self, save_to_history: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a sculpture by attaching shapes with PyTorch operations."""
        begin_section("Generating PyTorch Sculpture")

        try:
            if save_to_history:
                self._save_state_to_history()

            start_time = time.time()
            initial_memory = self.get_memory_usage()

            total_ops = (
                (1 if self.grid[0] == 1 else 0)
                + self.edges[0] + self.planes[0] + self.pipes[0]
            )
            current_op = 0
            log_info(f"Starting sculpture generation with {total_ops} operations")

            # --- Grid ---
            if self.grid[0] == 1:
                current_op += 1
                log_action(f"Adding grid ({current_op}/{total_ops})")
                self.structure, self.colors = attach_grid_pytorch(
                    self.structure, self.colors,
                    step=self.grid[1],
                    colors_dict=self.colors_dict,
                    device=self.device,
                    sparse_mode=self.sparse_mode,
                    verbose=self.verbose,
                )
                self._check_memory_and_optimize()

            # --- Edges ---
            for i in range(self.edges[0]):
                current_op += 1
                log_action(f"Adding edge {i+1}/{self.edges[0]} ({current_op}/{total_ops})")
                self.structure, self.colors = attach_edge_pytorch(
                    self.structure, self.colors,
                    element_edge_min_ratio=self.edges[1],
                    element_edge_max_ratio=self.edges[2],
                    step=self.step,
                    colors_dict=self.colors_dict,
                    device=self.device,
                    sparse_mode=self.sparse_mode,
                    verbose=self.verbose,
                )
                self._check_memory_and_optimize()

            # --- Planes ---
            for i in range(self.planes[0]):
                current_op += 1
                log_action(f"Adding plane {i+1}/{self.planes[0]} ({current_op}/{total_ops})")
                self.structure, self.colors = attach_plane_pytorch(
                    self.structure, self.colors,
                    element_plane_min_ratio=self.planes[1],
                    element_plane_max_ratio=self.planes[2],
                    step=self.step,
                    colors_dict=self.colors_dict,
                    device=self.device,
                    sparse_mode=self.sparse_mode,
                    verbose=self.verbose,
                )
                self._check_memory_and_optimize()

            # --- Pipes ---
            for i in range(self.pipes[0]):
                current_op += 1
                log_action(f"Adding pipe {i+1}/{self.pipes[0]} ({current_op}/{total_ops})")
                self.structure, self.colors = attach_pipe_pytorch(
                    self.structure, self.colors,
                    element_volume_min_ratio=self.pipes[1],
                    element_volume_max_ratio=self.pipes[2],
                    step=self.step,
                    colors_dict=self.colors_dict,
                    device=self.device,
                    sparse_mode=self.sparse_mode,
                    verbose=self.verbose,
                )
                self._check_memory_and_optimize()

            # --- Validate sculpture ---
            self._validate_sculpture_quality()

            # --- Stats ---
            generation_time = time.time() - start_time
            final_memory = self.get_memory_usage()

            if self.structure.is_sparse:
                dense = SparseTensorHandler.to_dense(self.structure)
                filled = torch.sum(dense > 0).item()
                total = dense.numel()
            else:
                filled = torch.sum(self.structure > 0).item()
                total = self.structure.numel()

            fill_pct = (filled / total) * 100

            self._generation_stats = {
                "generation_time": generation_time,
                "filled_voxels": filled,
                "total_voxels": total,
                "fill_percentage": fill_pct,
                "initial_memory_gb": initial_memory['allocated'],
                "final_memory_gb": final_memory['allocated'],
                "memory_delta_gb": final_memory['allocated'] - initial_memory['allocated'],
                "operations_count": total_ops,
                "sparse_mode": self.sparse_mode,
            }

            log_info(f"Filled voxels: {filled}/{total} ({fill_pct:.2f}%)")
            log_info(f"Memory usage: {final_memory['allocated']:.2f} GB")
            log_success(f"Sculpture generated in {generation_time:.2f} seconds")

            end_section()
            return self.structure, self.colors

        except Exception as e:
            log_error(f"Error generating sculpture: {str(e)}")
            end_section("Sculpture generation failed")
            raise

    def generate_architectural_sculpture(self, save_to_history: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a fixed scaffold with grid, three planes, and two orthogonal pipes."""
        begin_section("Generating Architectural PyTorch Sculpture")

        try:
            if save_to_history:
                self._save_state_to_history()

            start_time = time.time()
            initial_memory = self.get_memory_usage()

            total_ops = (1 if self.grid[0] == 1 else 0) + self.edges[0] + 3 + 2
            current_op = 0
            log_info(f"Starting architectural sculpture generation with {total_ops} operations")

            if self.grid[0] == 1:
                current_op += 1
                log_action(f"Adding grid ({current_op}/{total_ops})")
                self.structure, self.colors = attach_grid_pytorch(
                    self.structure, self.colors,
                    step=self.grid[1],
                    colors_dict=self.colors_dict,
                    device=self.device,
                    sparse_mode=self.sparse_mode,
                    verbose=self.verbose,
                )
                self._check_memory_and_optimize()

            # Compute slab z positions for snapping pipes
            floor_step = max(4, self.void_dim // 4)
            slab_zs_all = [0] + list(range(floor_step, self.void_dim - 1, floor_step))

            # No edges or planes in architectural mode — just grid, slabs, and pipes

            # Build list of gaps between slabs: (z_start, z_end) with 1 voxel clearance
            slab_zs = slab_zs_all + [self.void_dim]
            gaps = []
            for i in range(len(slab_zs) - 1):
                z_lo = slab_zs[i] + 1      # 1 voxel above lower slab
                z_hi = slab_zs[i + 1] - 1  # 1 voxel below upper slab
                if z_hi > z_lo:
                    gaps.append((z_lo, z_hi))

            # Slab is 80% of floor; pipes extend 1 voxel beyond slab edge
            slab_margin = max(1, self.void_dim // 10)
            pipe_start = max(0, slab_margin - 1)
            pipe_end = min(self.void_dim, self.void_dim - slab_margin + 1)

            # 3 pipes: red, blue, yellow — orthogonal, intercepting.
            # Two pipes share the same z-level but run x vs y → they intersect.
            # Third pipe at a different level for variety.
            pipe_colors_list = ["red", "blue", "yellow"]
            pipe_configs = []  # (axis, gap_start_idx, num_gaps)

            if len(gaps) >= 2:
                # Pipe 1 & 2: share a starting gap, orthogonal, different heights
                #   - One is 1 floor tall, other is 2 floors tall → they intersect
                # Pipe 3: at a completely different level
                shared_gap = random.randint(0, max(0, len(gaps) - 2))  # leave room for 2-floor pipe
                other_gaps = [i for i in range(len(gaps)) if i != shared_gap and i != shared_gap + 1]
                third_gap = random.choice(other_gaps) if other_gaps else 0
                pipe_configs = [
                    (0, shared_gap, 1),   # red: x-aligned, 1 floor tall
                    (1, shared_gap, 2 if shared_gap + 1 < len(gaps) else 1),  # blue: y-aligned, 2 floors tall → intersects red
                    (0, third_gap, 1),    # yellow: x-aligned, different level entirely
                ]
            elif len(gaps) >= 1:
                pipe_configs = [
                    (0, 0, 1),
                    (1, 0, 1),
                    (0, 0, 1),
                ]

            for pipe_idx, (axis_selection, gap_idx, num_gaps) in enumerate(pipe_configs):
                current_op += 1
                axis_name = "x" if axis_selection == 0 else "y"
                log_action(f"Adding {axis_name}-aligned pipe {pipe_idx+1}/3 ({current_op}/{total_ops})")

                # Compute z range spanning num_gaps consecutive gaps
                end_gap_idx = min(gap_idx + num_gaps - 1, len(gaps) - 1)
                z_lo = gaps[gap_idx][0]
                z_hi = gaps[end_gap_idx][1]
                snap_z = (z_lo + z_hi) // 2
                pipe_depth = z_hi - z_lo
                depth_ratio = pipe_depth / self.void_dim

                # Force specific color for this pipe
                pipe_color_override = {
                    "pipes": [pipe_colors_list[pipe_idx % len(pipe_colors_list)]]
                }

                # x/y span should be 50-80% of canvas (big like the yellow one)
                pipe_span_ratio = random.uniform(0.5, 0.8)
                self.structure, self.colors = attach_pipe_pytorch(
                    self.structure, self.colors,
                    element_volume_min_ratio=pipe_span_ratio,
                    element_volume_max_ratio=pipe_span_ratio,
                    step=self.step,
                    colors_dict=pipe_color_override,
                    device=self.device,
                    sparse_mode=self.sparse_mode,
                    wall_thickness=random.randint(1, 2),
                    axis_selection=axis_selection,
                    snap_z=snap_z,
                    snap_xy_range=(self.grid[1],) if self.grid[0] == 1 else None,
                    verbose=self.verbose,
                )
                self._check_memory_and_optimize()

            self._validate_sculpture_quality()

            generation_time = time.time() - start_time
            final_memory = self.get_memory_usage()

            if self.structure.is_sparse:
                dense = SparseTensorHandler.to_dense(self.structure)
                filled = torch.sum(dense > 0).item()
                total = dense.numel()
            else:
                filled = torch.sum(self.structure > 0).item()
                total = self.structure.numel()

            fill_pct = (filled / total) * 100
            self._generation_stats = {
                "generation_time": generation_time,
                "filled_voxels": filled,
                "total_voxels": total,
                "fill_percentage": fill_pct,
                "initial_memory_gb": initial_memory['allocated'],
                "final_memory_gb": final_memory['allocated'],
                "memory_delta_gb": final_memory['allocated'] - initial_memory['allocated'],
                "operations_count": total_ops,
                "sparse_mode": self.sparse_mode,
                "structure_mode": "architectural",
            }

            log_info(f"Filled voxels: {filled}/{total} ({fill_pct:.2f}%)")
            log_info(f"Memory usage: {final_memory['allocated']:.2f} GB")
            log_success(f"Architectural sculpture generated in {generation_time:.2f} seconds")

            end_section()
            return self.structure, self.colors

        except Exception as e:
            log_error(f"Error generating architectural sculpture: {str(e)}")
            end_section("Architectural sculpture generation failed")
            raise

    def _check_memory_and_optimize(self):
        """Check memory usage and optimize if necessary."""
        usage = self.get_memory_usage()
        if usage['allocated'] > self.memory_limit_gb:
            log_warning(f"Memory usage {usage['allocated']:.2f} GB exceeds limit")
            if not self.sparse_mode:
                sparsity = SparseTensorHandler.detect_sparsity(self.structure)
                if sparsity > self.sparse_threshold:
                    log_action("Auto-converting to sparse tensors")
                    self.to_sparse()

    def _validate_sculpture_quality(self):
        """Validate the quality of the generated sculpture."""
        begin_section("Validating sculpture quality")

        try:
            if self.structure.is_sparse:
                dense = SparseTensorHandler.to_dense(self.structure)
                filled = torch.sum(dense > 0).item()
            else:
                filled = torch.sum(self.structure > 0).item()

            if filled == 0:
                log_warning("Generated sculpture is empty")
                return False

            total = self.void_dim ** 3
            fill_pct = (filled / total) * 100

            if fill_pct < 1.0:
                log_warning(f"Very sparse sculpture: {fill_pct:.2f}% filled")
            elif fill_pct > 80.0:
                log_warning(f"Very dense sculpture: {fill_pct:.2f}% filled")
            else:
                log_info(f"Good sculpture density: {fill_pct:.2f}% filled")

            if self.structure.shape != self.colors.shape:
                log_error("Structure and colors tensors have inconsistent shapes")
                return False

            dense_struct = (
                SparseTensorHandler.to_dense(self.structure)
                if self.structure.is_sparse else self.structure
            )
            dense_colors = (
                SparseTensorHandler.to_dense(self.colors)
                if self.colors.is_sparse else self.colors
            )

            if torch.any(torch.isnan(dense_struct)) or torch.any(torch.isinf(dense_struct)):
                log_error("Structure tensor contains NaN/Inf values")
                return False

            if torch.any(torch.isnan(dense_colors)) or torch.any(torch.isinf(dense_colors)):
                log_error("Colors tensor contains NaN/Inf values")
                return False

            log_success("Sculpture quality validation passed")
            end_section()
            return True

        except Exception as e:
            log_error(f"Error validating sculpture quality: {str(e)}")
            end_section("Sculpture quality validation failed")
            return False

    # ================================
    # --- Statistics & Estimation ---
    # ================================

    def get_generation_stats(self) -> Dict[str, Any]:
        """Return statistics from the last sculpture generation."""
        return self._generation_stats.copy() if self._generation_stats else {}

    def estimate_generation_time(self) -> float:
        """Estimate sculpture generation time based on parameters."""
        base_times = {"grid": 0.1, "edge": 0.05, "plane": 0.1, "pipe": 0.15}
        t = 0.0
        if self.grid[0] == 1: t += base_times["grid"]
        t += self.edges[0] * base_times["edge"]
        t += self.planes[0] * base_times["plane"]
        t += self.pipes[0] * base_times["pipe"]
        if self.device == "cuda": t *= 0.7
        if self.sparse_mode: t *= 1.2
        size_factor = (self.void_dim / 64) ** 2
        return t * size_factor

    # ================================
    # --- Shape Operations ---
    # ================================

    def add_shape(self, shape_type: ShapeType, min_ratio=0.1, max_ratio=0.9,
                  save_to_history: bool = True, **kwargs) -> 'PyTorchSculptor':
        """Add a single shape to the sculpture."""
        begin_section(f"Adding {shape_type.name} to sculpture")
        try:
            if save_to_history: self._save_state_to_history()

            if shape_type == ShapeType.EDGE:
                self.structure, self.colors = attach_edge_pytorch(
                    self.structure, self.colors,
                    element_edge_min_ratio=min_ratio,
                    element_edge_max_ratio=max_ratio,
                    step=self.step, colors_dict=self.colors_dict,
                    device=self.device, sparse_mode=self.sparse_mode,
                    verbose=self.verbose, **kwargs)
            elif shape_type == ShapeType.PLANE:
                self.structure, self.colors = attach_plane_pytorch(
                    self.structure, self.colors,
                    element_plane_min_ratio=min_ratio,
                    element_plane_max_ratio=max_ratio,
                    step=self.step, colors_dict=self.colors_dict,
                    device=self.device, sparse_mode=self.sparse_mode,
                    verbose=self.verbose, **kwargs)
            elif shape_type == ShapeType.PIPE:
                self.structure, self.colors = attach_pipe_pytorch(
                    self.structure, self.colors,
                    element_volume_min_ratio=min_ratio,
                    element_volume_max_ratio=max_ratio,
                    step=self.step, colors_dict=self.colors_dict,
                    device=self.device, sparse_mode=self.sparse_mode,
                    verbose=self.verbose, **kwargs)
            else:
                raise ValueError(f"Unsupported shape type: {shape_type}")

            self._check_memory_and_optimize()
            log_success(f"{shape_type.name} added successfully")
            end_section()
            return self
        except Exception as e:
            log_error(f"Error adding shape: {str(e)}")
            end_section("Shape addition failed")
            raise

    def reset(self, save_to_history=True) -> 'PyTorchSculptor':
        """Reset sculpture to empty structure."""
        begin_section("Resetting PyTorch Sculpture")
        try:
            if save_to_history: self._save_state_to_history()
            self._initialize_tensors()
            self._generation_stats = {}
            log_success("Sculpture reset successfully")
            end_section()
            return self
        except Exception as e:
            log_error(f"Error resetting sculpture: {str(e)}")
            end_section("Sculpture reset failed")
            raise

    # ================================
    # --- Save / Load ---
    # ================================

    def save(self, directory="output", filename_prefix="pytorch_sculpture",
             save_structure=True, save_colors=True, save_metadata=True,
             compress=True) -> Dict[str, str]:
        """Save sculpture tensors and metadata."""
        begin_section("Saving Sculpture")
        try:
            os.makedirs(directory, exist_ok=True)
            saved = {}
            ts = time.strftime("%Y%m%d-%H%M%S")

            if save_structure:
                path = os.path.join(directory, f"{filename_prefix}_structure_{ts}.pt")
                struct_to_save = (
                    SparseTensorHandler.to_dense(self.structure)
                    if self.structure.is_sparse else self.structure
                )
                torch.save(struct_to_save, path, _use_new_zipfile_serialization=compress)
                saved["structure"] = path
                log_success(f"Saved structure to {path}")

            if save_colors:
                path = os.path.join(directory, f"{filename_prefix}_colors_{ts}.pt")
                colors_to_save = (
                    SparseTensorHandler.to_dense(self.colors)
                    if self.colors.is_sparse else self.colors
                )
                torch.save(colors_to_save, path, _use_new_zipfile_serialization=compress)
                saved["colors"] = path
                log_success(f"Saved colors to {path}")

            if save_metadata:
                path = os.path.join(directory, f"{filename_prefix}_metadata_{ts}.pt")
                metadata = {
                    "void_dim": self.void_dim,
                    "edges": self.edges, "planes": self.planes, "pipes": self.pipes,
                    "grid": self.grid, "colors_dict": self.colors_dict,
                    "step": self.step, "device": self.device,
                    "sparse_mode": self.sparse_mode,
                    "sparse_threshold": self.sparse_threshold,
                    "memory_limit_gb": self.memory_limit_gb,
                    "tensor_info": self.get_tensor_info(),
                    "generation_stats": self.get_generation_stats(),
                    "timestamp": ts,
                }
                torch.save(metadata, path)
                saved["metadata"] = path
                log_success(f"Saved metadata to {path}")

            log_success("Sculpture saved successfully")
            end_section()
            return saved
        except Exception as e:
            log_error(f"Error saving sculpture: {str(e)}")
            end_section("Save failed")
            raise

    @classmethod
    def load(cls, structure_path: str, colors_path: Optional[str] = None,
             metadata_path: Optional[str] = None, device="auto",
             verbose=False) -> 'PyTorchSculptor':
        """Load sculpture from files."""
        begin_section("Loading Sculpture")
        try:
            # Handle device mapping
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            structure = torch.load(structure_path, map_location=device)
            log_success(f"Loaded structure from {structure_path}")
            colors = (torch.load(colors_path, map_location=device)
                      if colors_path else torch.zeros_like(structure, dtype=torch.int16))
            if not colors_path: log_info("Created empty colors tensor")

            if metadata_path:
                metadata = torch.load(metadata_path, map_location="cpu")
                sculptor = cls(
                    void_dim=metadata.get("void_dim", structure.shape[0]),
                    edges=metadata.get("edges", (1, 0.3, 0.5)),
                    planes=metadata.get("planes", (1, 0.3, 0.5)),
                    pipes=metadata.get("pipes", (1, 0.3, 0.5)),
                    grid=metadata.get("grid", (1, 4)),
                    colors=metadata.get("colors_dict"),
                    step=metadata.get("step", 1),
                    device=device,
                    sparse_mode=metadata.get("sparse_mode", False),
                    sparse_threshold=metadata.get("sparse_threshold", 0.1),
                    memory_limit_gb=metadata.get("memory_limit_gb", 4.0),
                    verbose=verbose,
                )
            else:
                sculptor = cls(void_dim=structure.shape[0], device=device, verbose=verbose)
                log_info("No metadata file provided, using defaults")

            sculptor.structure = structure.to(sculptor.device)
            sculptor.colors = colors.to(sculptor.device)
            sculptor.validate_configuration()
            log_success("Sculpture loaded successfully")
            end_section()
            return sculptor
        except Exception as e:
            log_error(f"Error loading sculpture: {str(e)}")
            end_section("Load failed")
            raise

    # ================================
    # --- Visualization & Clone ---
    # ================================

    def visualize(self, title="PyTorch 3D Sculpture", hide_axis=True,
                  save_path=None, save_array=False, save_dir="output"):
        """Visualize the generated sculpture."""
        begin_section("Visualizing PyTorch Sculpture")
        try:
            from visualization import Visualizer
            visualizer = Visualizer(figsize=15, dpi=100)
            
            if self.structure.is_sparse:
                structure_np = SparseTensorHandler.to_dense(self.structure).cpu().numpy()
                colors_np = SparseTensorHandler.to_dense(self.colors).cpu().numpy()
            else:
                structure_np = self.structure.cpu().numpy()
                colors_np = self.colors.cpu().numpy()
            
            if save_path or save_array:
                os.makedirs(save_dir, exist_ok=True)
            
            fig = visualizer.plot_sculpture(
                structure=structure_np, colors=colors_np, title=title,
                hide_axis=hide_axis, save_path=save_path,
                save_array=save_array, save_dir=save_dir,
            )
            
            log_success("PyTorch sculpture visualization completed")
            end_section()
            return fig
        except Exception as e:
            log_error(f"Error visualizing sculpture: {str(e)}")
            end_section("Visualization failed")
            raise

    def clone(self) -> 'PyTorchSculptor':
        """Create a deep copy of this sculptor."""
        begin_section("Cloning PyTorch Sculptor")
        try:
            new_sculptor = PyTorchSculptor(
                void_dim=self.void_dim, edges=self.edges, planes=self.planes,
                pipes=self.pipes, grid=self.grid, colors=copy.deepcopy(self.colors_dict),
                step=self.step, device=self.device, sparse_mode=self.sparse_mode,
                sparse_threshold=self.sparse_threshold, memory_limit_gb=self.memory_limit_gb,
                verbose=self.verbose,
            )
            new_sculptor.structure = self.structure.clone()
            new_sculptor.colors = self.colors.clone()
            new_sculptor._generation_stats = self._generation_stats.copy()
            log_success("PyTorch sculptor cloned successfully")
            end_section()
            return new_sculptor
        except Exception as e:
            log_error(f"Error cloning sculptor: {str(e)}")
            end_section("Cloning failed")
            raise

    def clear_history(self) -> 'PyTorchSculptor':
        """Clear the undo/redo history to free memory."""
        self._history.clear()
        self._history_index = -1
        log_info("History cleared")
        return self

    def get_history_info(self) -> Dict[str, Any]:
        """Get information about the current history state."""
        return {
            "history_size": len(self._history),
            "current_index": self._history_index,
            "can_undo": self._history_index > 0,
            "can_redo": self._history_index < len(self._history) - 1,
            "max_history_size": self._max_history_size,
        }

    def undo(self) -> 'PyTorchSculptor':
        """Undo the last operation by restoring from history."""
        begin_section("Undoing last operation")
        try:
            if self._history_index <= 0:
                log_warning("No operations to undo")
                end_section("Nothing to undo")
                return self
            
            self._history_index -= 1
            state = self._history[self._history_index]
            self.structure = state["structure"].clone()
            self.colors = state["colors"].clone()
            self.sparse_mode = state["sparse_mode"]
            
            log_success("Operation undone successfully")
            end_section()
            return self
        except Exception as e:
            log_error(f"Error undoing operation: {str(e)}")
            end_section("Undo failed")
            raise

    def redo(self) -> 'PyTorchSculptor':
        """Redo the next operation by moving forward in history."""
        begin_section("Redoing next operation")
        try:
            if self._history_index >= len(self._history) - 1:
                log_warning("No operations to redo")
                end_section("Nothing to redo")
                return self
            
            self._history_index += 1
            state = self._history[self._history_index]
            self.structure = state["structure"].clone()
            self.colors = state["colors"].clone()
            self.sparse_mode = state["sparse_mode"]
            
            log_success("Operation redone successfully")
            end_section()
            return self
        except Exception as e:
            log_error(f"Error redoing operation: {str(e)}")
            end_section("Redo failed")
            raise
        """Visualize the sculpture using visualization.py."""
        begin_section("Visualizing Sculpture")
        try:
            from visualization import Visualizer
            vis = Visualizer(figsize=15, dpi=100)
            struct_np = (SparseTensorHandler.to_dense(self.structure).cpu().numpy()
                         if self.structure.is_sparse else self.structure.cpu().numpy())
            colors_np = (SparseTensorHandler.to_dense(self.colors).cpu().numpy()
                         if self.colors.is_sparse else self.colors.cpu().numpy())
            if save_path or save_array: os.makedirs(save_dir, exist_ok=True)
            fig = vis.plot_sculpture(struct_np, colors_np, title=title,
                                     hide_axis=hide_axis, save_path=save_path,
                                     save_array=save_array, save_dir=save_dir)
            log_success("Visualization completed")
            end_section()
            return fig
        except Exception as e:
            log_error(f"Error visualizing sculpture: {str(e)}")
            end_section("Visualization failed")
            raise

    def clone(self) -> 'PyTorchSculptor':
        """Create a deep copy of this sculptor."""
        begin_section("Cloning Sculpture")
        try:
            new = PyTorchSculptor(
                void_dim=self.void_dim, edges=self.edges, planes=self.planes,
                pipes=self.pipes, grid=self.grid,
                colors=copy.deepcopy(self.colors_dict),
                step=self.step, device=self.device,
                sparse_mode=self.sparse_mode,
                sparse_threshold=self.sparse_threshold,
                memory_limit_gb=self.memory_limit_gb,
                verbose=self.verbose,
            )
            new.structure = self.structure.clone()
            new.colors = self.colors.clone()
            new._generation_stats = self._generation_stats.copy()
            log_success("Clone created")
            end_section()
            return new
        except Exception as e:
            log_error(f"Error cloning sculpture: {str(e)}")
            end_section("Clone failed")
            raise


# ================================
# --- Factory & Demo ---
# ================================

def create_pytorch_sculptor(**kwargs) -> PyTorchSculptor:
    """Factory function for backward compatibility."""
    return PyTorchSculptor(**kwargs)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from datetime import datetime

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    begin_section("PyTorch DeepSculpt Demo")
    log_info("Creating a 3D sculpture demo with PyTorch")

    sculptor = PyTorchSculptor(
        void_dim=32, edges=(2, 0.2, 0.6), planes=(2, 0.3, 0.7),
        pipes=(1, 0.4, 0.7), grid=(1, 4),
        device="auto", sparse_mode=False, verbose=True,
    )

    log_info(f"Created sculptor: {sculptor}")
    log_action("Generating sculpture")
    struct, colors = sculptor.generate_sculpture()

    stats = sculptor.get_generation_stats()
    info = sculptor.get_tensor_info()
    log_info(f"Generation time: {stats.get('generation_time', 0):.2f} sec")
    log_info(f"Fill percentage: {stats.get('fill_percentage', 0):.2f}%")
    log_info(f"Memory: {info['memory_usage']['allocated']:.2f} GB")
    log_info(f"Sparsity: {info['sparsity']:.3f}")

    fig_path = os.path.join(output_dir, f"pytorch_sculpture_{timestamp}.png")
    sculptor.visualize(title="PyTorch DeepSculpt", save_path=fig_path,
                       save_array=True, save_dir=output_dir)
    log_success(f"Saved visualization to {fig_path}")

    saved = sculptor.save(directory=output_dir,
                          filename_prefix=f"pytorch_sculpture_{timestamp}")
    for k, v in saved.items():
        log_success(f"Saved {k} to {v}")

    cloned = sculptor.clone()
    log_info(f"Cloned sculptor: {cloned}")

    log_success("PyTorch DeepSculpt demo completed")
    end_section()
