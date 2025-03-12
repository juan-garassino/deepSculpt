"""
Utility Functions for DeepSculpt Geometry Operations
This module provides common functions for 3D shape manipulation, coordinate handling,
random geometry generation, and debugging utilities used throughout the DeepSculpt package.

Key features:
- Random dimension generation: Functions for creating size parameters within constraints
- Coordinate operations: Functions for position selection and validation
- Shape insertion: Utilities for adding shapes to 3D arrays
- Validation: Functions for checking geometric constraints and boundaries
- Debug utilities: Tools for inspecting and reporting on 3D structures

Dependencies:
- logger.py: For detailed operation logging
- numpy: For array manipulation and mathematical operations

Used by:
- shapes.py: For coordinate operations and validation during shape creation
- visualization.py: For data transformation before visualization
- sculptor.py: For geometry validation and manipulation
- collector.py: For array operations during dataset generation
- curator.py: For data inspection and validation

TODO:
- Add support for rotated shape insertion
- Implement more efficient array operations for large voids
- Add topological analysis functions
- Improve error detection and correction for invalid geometries
- Add support for non-cubic voids and irregular grids
"""

import numpy as np
import random
from typing import Tuple, List, Optional, Any, Dict
from logger import log_info, log_error, log_warning, begin_section, end_section


def return_axis(
    void: np.ndarray, color_void: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Selects a random plane from a 3D numpy array along a random axis.

    Args:
        void: The 3D numpy array to select a plane from.
        color_void: The 3D numpy array that holds the color information.

    Returns:
        A tuple containing:
            - working_plane: The randomly selected plane.
            - color_parameters: The color information of the selected plane.
            - section: The index of the selected plane.
    """
    section = np.random.randint(low=0, high=void.shape[0])
    axis_selection = np.random.randint(low=0, high=3)

    log_info(f"Selected axis {axis_selection}, section {section}", is_last=False)

    if axis_selection == 0:
        working_plane = void[section, :, :]
        color_parameters = color_void[section, :, :]
    elif axis_selection == 1:
        working_plane = void[:, section, :]
        color_parameters = color_void[:, section, :]
    elif axis_selection == 2:
        working_plane = void[:, :, section]
        color_parameters = color_void[:, :, section]
    else:
        log_error("Axis selection value out of range.")
        raise ValueError("Axis selection value out of range.")

    return working_plane, color_parameters, section


def generate_random_size(
    min_ratio: float, max_ratio: float, base_size: int, step: int = 1
) -> int:
    """
    Generate a random size based on given ratios and base size.

    Args:
        min_ratio: Minimum size ratio relative to base_size
        max_ratio: Maximum size ratio relative to base_size
        base_size: Reference size (usually the smallest dimension of the void)
        step: Step size for the random range

    Returns:
        Integer representing the random size
    """
    min_size = max(int(min_ratio * base_size), 2)  # Ensure minimum size of 2
    max_size = max(int(max_ratio * base_size), min_size + 1)  # Ensure max > min

    if step > 1:
        # Adjust to be multiples of step
        min_size = (min_size // step) * step
        max_size = (max_size // step) * step
        if min_size == max_size:
            return min_size

    return random.randrange(min_size, max_size, step)


def select_random_position(max_pos: int, size: int) -> int:
    """
    Select a random position to insert a shape within bounds.

    Args:
        max_pos: Maximum position value (usually dimension size)
        size: Size of the shape to be inserted

    Returns:
        Integer representing the random position
    """
    return random.randint(0, max(0, max_pos - size))


def insert_shape(
    void: np.ndarray, shape_indices: tuple, values: np.ndarray = None
) -> np.ndarray:
    """
    Insert a shape into the void at the given indices.

    Args:
        void: 3D NumPy array representing the space
        shape_indices: Tuple of slices or indices where to insert the shape
        values: Values to insert, if None uses 1s

    Returns:
        Updated void array with the shape inserted
    """
    if values is None:
        void[shape_indices] = 1
    else:
        void[shape_indices] = values
    return void


def assign_color(
    color_void: np.ndarray, shape_indices: tuple, color: Any
) -> np.ndarray:
    """
    Assign color to the shape in the color void array.

    Args:
        color_void: 3D NumPy array of objects representing colors
        shape_indices: Tuple of slices or indices where the shape is
        color: Color to assign to the shape

    Returns:
        Updated color_void array with colors assigned to the shape
    """
    color_void[shape_indices] = color
    return color_void


def validate_dimensions(shape_size: List[int], void_shape: Tuple[int, ...]) -> bool:
    """
    Validate that the shape fits within the void dimensions.

    Args:
        shape_size: Dimensions of the shape
        void_shape: Dimensions of the void

    Returns:
        Boolean indicating if the shape fits in the void
    """
    return all(s <= v for s, v in zip(shape_size, void_shape))


def validate_bounds(
    start_pos: List[int], shape_size: List[int], void_shape: Tuple[int, ...]
) -> bool:
    """
    Validate that the shape at the given position fits within the void bounds.

    Args:
        start_pos: Starting position coordinates
        shape_size: Dimensions of the shape
        void_shape: Dimensions of the void

    Returns:
        Boolean indicating if the shape at the position fits in the void
    """
    for i in range(len(start_pos)):
        if start_pos[i] < 0 or start_pos[i] + shape_size[i] > void_shape[i]:
            return False
    return True


def select_random_color(colors: List[str]) -> str:
    """
    Select a random color from a list or return the color if it's a string.

    Args:
        colors: List of color strings or a single color string

    Returns:
        Selected color string
    """
    if isinstance(colors, list):
        return random.choice(colors)
    return colors


def create_debug_info(void: np.ndarray, filled_only: bool = True) -> Dict[str, Any]:
    """
    Create debug information about the void array.

    Args:
        void: The 3D numpy array
        filled_only: If True, only count filled voxels

    Returns:
        Dictionary with debug information
    """
    info = {
        "shape": void.shape,
        "total_voxels": void.size,
    }

    if filled_only:
        filled = void > 0
        info["filled_voxels"] = np.sum(filled)
        info["fill_percentage"] = (info["filled_voxels"] / info["total_voxels"]) * 100

    return info


def print_debug_info(info: Dict[str, Any]):
    """
    Print debug information in a structured format.

    Args:
        info: Dictionary with debug information
    """
    begin_section("Debug Information")

    for key, value in info.items():
        if key == "fill_percentage":
            log_info(f"{key}: {value:.2f}%")
        else:
            log_info(f"{key}: {value}")

    end_section()
