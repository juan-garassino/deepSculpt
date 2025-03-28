"""
3D Shape Generation System for DeepSculpt
This module provides comprehensive functionality for creating various 3D voxel-based shapes
including edges (1D), planes (2D), pipes (hollow 3D), and complex grid structures.
It handles random geometry generation with configurable constraints.

Key features:
- Shape types: Support for edges, planes, pipes, and grid structures
- Parameterized generation: Flexible size and position constraints
- Random variation: Controlled randomness for diverse shape creation
- Color assignment: Automatic color mapping for visualization
- Component attachment: Functions for adding shapes to existing structures

Dependencies:
- logger.py: For process tracking and debugging
- utils.py: For coordinate operations and validation
- numpy: For array operations and random number generation
- enum: For shape type categorization

Used by:
- sculptor.py: For building complete 3D sculptures
- collector.py: For batch generation of training data
- visualization.py: For creating sample shapes to visualize

Terminology:
- structure: 3D numpy array representing the sculpture shape (formerly "void")
- colors: 3D numpy array with color information (formerly "color_void")

TODO:
- Add support for curved and organic shapes
- Implement more complex geometric primitives
- Add shape intersection and boolean operations
- Improve performance for large shape generation
- Support custom shape templates and patterns
- Add procedural texture and material properties
"""

import numpy as np
import random
from enum import Enum
from typing import Tuple, List, Dict, Any, Optional, Union
from logger import (
    begin_section,
    end_section,
    log_action,
    log_success,
    log_error,
    log_info,
    log_warning,
)
from utils import (
    generate_random_size,
    select_random_position,
    insert_shape,
    assign_color,
    validate_dimensions,
    validate_bounds,
    select_random_color,
    return_axis,
)


class ShapeType(Enum):
    """Enumeration of different shape types."""

    EDGE = 1  # 1D linear shape
    PLANE = 2  # 2D planar shape
    PIPE = 3  # 3D hollow shape (box with empty interior)
    VOLUME = 4  # 3D solid shape (filled box)


def attach_edge(
    structure: np.ndarray,
    colors: np.ndarray,
    element_edge_min_ratio: float = 0.1,
    element_edge_max_ratio: float = 0.9,
    step: int = 1,
    colors_dict: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Attach an edge (1D line) to the structure.

    Args:
        structure: 3D array representing the structure space
        colors: 3D array with color information
        element_edge_min_ratio: Minimum edge size ratio
        element_edge_max_ratio: Maximum edge size ratio
        step: Step size for edge dimensions
        colors_dict: Dictionary of colors for different shape types
        verbose: Whether to print detailed information

    Returns:
        Updated structure and colors arrays
    """
    begin_section("Attaching edge (1D line)")

    try:
        if colors_dict is None:
            colors_dict = {"edges": "red"}

        # Select a random axis (0, 1, or 2)
        axis = random.randint(0, 2)
        log_info(f"Selected axis: {axis}")

        # Generate edge size
        edge_size = generate_random_size(
            element_edge_min_ratio, element_edge_max_ratio, structure.shape[axis], step
        )
        log_info(f"Edge size: {edge_size}")

        # Select edge position
        edge_position = select_random_position(structure.shape[axis], edge_size)
        log_info(f"Edge position on axis {axis}: {edge_position}")

        # Create the position indices for the edge
        position = [0, 0, 0]
        position[axis] = slice(edge_position, edge_position + edge_size)

        # For the other two axes, select random positions
        other_axes = [i for i in range(3) if i != axis]
        for i in other_axes:
            position[i] = random.randint(0, structure.shape[i] - 1)

        # Log position information
        pos_info = [
            f"{i}: {'slice' if isinstance(p, slice) else p}"
            for i, p in enumerate(position)
        ]
        log_info(f"Edge position indices: {pos_info}")

        # Select color
        edge_color = select_random_color(colors_dict["edges"])
        log_info(f"Selected color: {edge_color}")

        # Insert the edge into the structure
        structure[tuple(position)] = 1

        # Assign color
        colors[tuple(position)] = edge_color

        log_success("Edge attached successfully")
        end_section()

        return structure.astype("int8"), colors

    except Exception as e:
        log_error(f"Error attaching edge: {str(e)}")
        end_section("Edge attachment failed")
        raise


def attach_plane(
    structure: np.ndarray,
    colors: np.ndarray,
    element_plane_min_ratio: float = 0.1,
    element_plane_max_ratio: float = 0.9,
    step: int = 1,
    colors_dict: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Attach a plane (2D surface) to the structure.

    Args:
        structure: 3D array representing the structure space
        colors: 3D array with color information
        element_plane_min_ratio: Minimum plane size ratio
        element_plane_max_ratio: Maximum plane size ratio
        step: Step size for plane dimensions
        colors_dict: Dictionary of colors for different shape types
        verbose: Whether to print detailed information

    Returns:
        Updated structure and colors arrays
    """
    begin_section("Attaching plane (2D surface)")

    try:
        if colors_dict is None:
            colors_dict = {"planes": "green"}

        # Get a random working plane and its color parameters
        working_plane, color_parameters, section = return_axis(structure, colors)
        log_info(f"Working on plane with shape {working_plane.shape}")

        # Calculate size constraints
        element_plane_min_index = int(element_plane_min_ratio * structure.shape[0])
        element_plane_max = int(element_plane_max_ratio * structure.shape[0])

        # Create the element to be inserted (a rectangular plane)
        element = np.ones(
            (
                generate_random_size(
                    element_plane_min_ratio,
                    element_plane_max_ratio,
                    structure.shape[0],
                    step,
                ),
                generate_random_size(
                    element_plane_min_ratio,
                    element_plane_max_ratio,
                    structure.shape[1],
                    step,
                ),
            )
        )

        log_info(f"Created plane element with shape {element.shape}")

        # Find the delta between working plane and element sizes
        delta = np.array(working_plane.shape) - np.array(element.shape)

        if np.any(delta < 0):
            log_warning("Plane too large for working plane, skipping")
            end_section("Plane attachment skipped")
            return structure, colors

        # Find random position for top-left corner
        top_left_corner = np.array(
            (
                np.random.randint(low=0, high=delta[0] + 1),
                np.random.randint(low=0, high=delta[1] + 1),
            )
        )

        # Calculate bottom right corner
        bottom_right_corner = top_left_corner + np.array(element.shape)

        log_info(
            f"Placing plane at position {top_left_corner} to {bottom_right_corner}"
        )

        # Select color
        plane_color = select_random_color(colors_dict["planes"])
        log_info(f"Selected color: {plane_color}")

        # Insert the plane into the working plane
        working_plane[
            top_left_corner[0] : bottom_right_corner[0],
            top_left_corner[1] : bottom_right_corner[1],
        ] = element

        # Assign color
        color_parameters[
            top_left_corner[0] : bottom_right_corner[0],
            top_left_corner[1] : bottom_right_corner[1],
        ] = plane_color

        log_success("Plane attached successfully")
        end_section()

        return structure.astype("int8"), colors

    except Exception as e:
        log_error(f"Error attaching plane: {str(e)}")
        end_section("Plane attachment failed")
        raise


def attach_pipe(
    structure: np.ndarray,
    colors: np.ndarray,
    element_volume_min_ratio: float = 0.1,
    element_volume_max_ratio: float = 0.9,
    step: int = 1,
    colors_dict: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Attach a pipe (hollow 3D structure) to the structure.

    Args:
        structure: 3D array representing the structure space
        colors: 3D array with color information
        element_volume_min_ratio: Minimum volume size ratio
        element_volume_max_ratio: Maximum volume size ratio
        step: Step size for volume dimensions
        colors_dict: Dictionary of colors for different shape types
        verbose: Whether to print detailed information

    Returns:
        Updated structure and colors arrays
    """
    begin_section("Attaching pipe (hollow 3D structure)")

    try:
        if colors_dict is None:
            colors_dict = {"pipes": ["blue", "cyan", "magenta"]}

        # Generate random dimensions for the pipe
        width = generate_random_size(
            element_volume_min_ratio, element_volume_max_ratio, structure.shape[0], step
        )
        height = generate_random_size(
            element_volume_min_ratio, element_volume_max_ratio, structure.shape[1], step
        )
        depth = generate_random_size(
            element_volume_min_ratio, element_volume_max_ratio, structure.shape[2], step
        )

        log_info(f"Pipe dimensions: width={width}, height={height}, depth={depth}")

        # Check if dimensions fit within structure
        if not validate_dimensions([width, height, depth], structure.shape):
            log_warning("Pipe dimensions too large for structure, skipping")
            end_section("Pipe attachment skipped")
            return structure, colors

        # Select random position for the pipe
        x_pos = select_random_position(structure.shape[0], width)
        y_pos = select_random_position(structure.shape[1], height)
        z_pos = select_random_position(structure.shape[2], depth)

        log_info(f"Pipe position: x={x_pos}, y={y_pos}, z={z_pos}")

        # Select random design parameters
        axis_selection = np.random.randint(low=0, high=2)
        shape_selection = np.random.randint(low=0, high=2)

        log_info(
            f"Design parameters: axis_selection={axis_selection}, shape_selection={shape_selection}"
        )

        # Select color
        pipe_color = select_random_color(colors_dict["pipes"])
        log_info(f"Selected color: {pipe_color}")

        # Create the corner coordinates
        corner_1 = np.array((x_pos, y_pos, z_pos))
        corner_2 = np.array((x_pos + width, y_pos, z_pos))
        corner_3 = np.array((x_pos, y_pos, z_pos + depth))
        corner_4 = np.array((x_pos + width, y_pos, z_pos + depth))
        corner_5 = np.array((x_pos, y_pos + height, z_pos))
        corner_6 = np.array((x_pos + width, y_pos + height, z_pos))
        corner_7 = np.array((x_pos, y_pos + height, z_pos + depth))
        corner_8 = np.array((x_pos + width, y_pos + height, z_pos + depth))

        # Create floor and ceiling
        structure[x_pos : x_pos + width, y_pos : y_pos + height, z_pos] = 1
        colors[x_pos : x_pos + width, y_pos : y_pos + height, z_pos] = pipe_color

        structure[x_pos : x_pos + width, y_pos : y_pos + height, z_pos + depth - 1] = 1
        colors[
            x_pos : x_pos + width, y_pos : y_pos + height, z_pos + depth - 1
        ] = pipe_color

        # Create walls based on design parameters
        if shape_selection == 0:
            if axis_selection == 0:
                # First wall
                structure[x_pos, y_pos : y_pos + height, z_pos : z_pos + depth] = 1
                colors[
                    x_pos, y_pos : y_pos + height, z_pos : z_pos + depth
                ] = pipe_color

                # Second wall
                structure[
                    x_pos + width - 1, y_pos : y_pos + height, z_pos : z_pos + depth
                ] = 1
                colors[
                    x_pos + width - 1, y_pos : y_pos + height, z_pos : z_pos + depth
                ] = pipe_color
            else:
                # First wall
                structure[x_pos : x_pos + width, y_pos, z_pos : z_pos + depth] = 1
                colors[x_pos : x_pos + width, y_pos, z_pos : z_pos + depth] = pipe_color

                # Second wall
                structure[
                    x_pos : x_pos + width, y_pos + height - 1, z_pos : z_pos + depth
                ] = 1
                colors[
                    x_pos : x_pos + width, y_pos + height - 1, z_pos : z_pos + depth
                ] = pipe_color
        else:
            if axis_selection == 0:
                # First wall
                structure[x_pos, y_pos : y_pos + height, z_pos : z_pos + depth] = 1
                colors[
                    x_pos, y_pos : y_pos + height, z_pos : z_pos + depth
                ] = pipe_color

                # Second wall
                structure[x_pos : x_pos + width, y_pos, z_pos : z_pos + depth] = 1
                colors[x_pos : x_pos + width, y_pos, z_pos : z_pos + depth] = pipe_color
            else:
                # First wall
                structure[
                    x_pos + width - 1, y_pos : y_pos + height, z_pos : z_pos + depth
                ] = 1
                colors[
                    x_pos + width - 1, y_pos : y_pos + height, z_pos : z_pos + depth
                ] = pipe_color

                # Second wall
                structure[x_pos : x_pos + width, y_pos, z_pos : z_pos + depth] = 1
                colors[x_pos : x_pos + width, y_pos, z_pos : z_pos + depth] = pipe_color

        log_success("Pipe attached successfully")
        end_section()

        return structure.astype("int8"), colors

    except Exception as e:
        log_error(f"Error attaching pipe: {str(e)}")
        end_section("Pipe attachment failed")
        raise


def attach_grid(
    structure: np.ndarray,
    colors: np.ndarray,
    step: int = 1,
    colors_dict: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Attach a grid structure to the structure.

    Args:
        structure: 3D array representing the structure space
        colors: 3D array with color information
        step: Step size for grid spacing
        colors_dict: Dictionary of colors for different shape types
        verbose: Whether to print detailed information

    Returns:
        Updated structure and colors arrays
    """
    begin_section("Attaching grid structure")

    try:
        if colors_dict is None:
            colors_dict = {"edges": "red"}

        structure_dim = structure.shape[0]
        log_info(f"Creating grid in structure of dimension {structure_dim}")

        # Calculate grid positions
        locations = []

        if structure_dim % 2 == 0:
            left_position = structure_dim / 2 - (step / 2)
            right_position = left_position + (step + 1)
            locations.append(int(left_position))
            locations.append(int(right_position))

            while left_position > 0 and right_position < (structure_dim - (step + 1)):
                left_position = left_position - (step + 1)
                right_position = right_position + (step + 1)
                locations.append(int(left_position))
                locations.append(int(right_position))

        # Sort and adjust locations
        X = np.array(sorted(locations))
        Y = np.array(sorted(locations))

        log_info(f"Grid positions X: {X}")
        log_info(f"Grid positions Y: {Y}")

        # Generate random heights for grid columns
        Z = np.random.randint(
            low=structure_dim // 4, high=structure_dim // 2, size=(len(X), len(Y))
        )

        # Create grid columns
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                height = Z[i, j]
                structure[x, y, 0:height] = 1
                colors[x, y, 0:height] = select_random_color(colors_dict["edges"])
                log_info(
                    f"Created column at x={x}, y={y} with height={height}",
                    is_last=(i == len(X) - 1 and j == len(Y) - 1),
                )

        # Create a base (floor)
        structure[:, :, 0] = 1
        colors[structure[:, :, 0] == 1] = select_random_color(colors_dict["edges"])

        log_success(f"Grid created with {len(X) * len(Y)} columns")
        end_section()

        return structure.astype("int8"), colors

    except Exception as e:
        log_error(f"Error attaching grid: {str(e)}")
        end_section("Grid attachment failed")
        raise


def attach_shape(
    structure: np.ndarray,
    colors: np.ndarray,
    shape_type: ShapeType,
    min_ratio: float = 0.1,
    max_ratio: float = 0.9,
    step: int = 1,
    colors_dict: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generic function to attach any type of shape to the structure.

    Args:
        structure: 3D array representing the structure space
        colors: 3D array with color information
        shape_type: Type of shape to attach (from ShapeType enum)
        min_ratio: Minimum size ratio
        max_ratio: Maximum size ratio
        step: Step size for dimensions
        colors_dict: Dictionary of colors for different shape types
        verbose: Whether to print detailed information

    Returns:
        Updated structure and colors arrays
    """
    begin_section(f"Attaching {shape_type.name.lower()} shape")

    try:
        if colors_dict is None:
            colors_dict = {
                "edges": "red",
                "planes": "green",
                "pipes": ["blue", "cyan", "magenta"],
                "volumes": ["purple", "brown", "orange"],
            }

        # Call the appropriate shape function based on type
        if shape_type == ShapeType.EDGE:
            result = attach_edge(
                structure, colors, min_ratio, max_ratio, step, colors_dict, verbose
            )
        elif shape_type == ShapeType.PLANE:
            result = attach_plane(
                structure, colors, min_ratio, max_ratio, step, colors_dict, verbose
            )
        elif shape_type == ShapeType.PIPE:
            result = attach_pipe(
                structure, colors, min_ratio, max_ratio, step, colors_dict, verbose
            )
        elif shape_type == ShapeType.VOLUME:
            # For volume, we could implement a solid 3D shape
            log_warning("Volume attachment not implemented, using pipe instead")
            result = attach_pipe(
                structure, colors, min_ratio, max_ratio, step, colors_dict, verbose
            )
        else:
            log_error(f"Unknown shape type: {shape_type}")
            end_section("Shape attachment failed")
            raise ValueError(f"Unknown shape type: {shape_type}")

        log_success(f"{shape_type.name.capitalize()} attached successfully")
        end_section()

        return result

    except Exception as e:
        log_error(f"Error in attach_shape: {str(e)}")
        end_section("Shape attachment failed")
        raise


# Example usage when run as main script
if __name__ == "__main__":
    # Create a structure and colors
    structure_dim = 20
    structure = np.zeros((structure_dim, structure_dim, structure_dim))
    colors = np.empty(structure.shape, dtype=object)

    # Define colors dictionary
    colors_dict = {
        "edges": "red",
        "planes": "green",
        "pipes": ["blue", "cyan", "magenta"],
    }

    # Test edge attachment
    structure, colors = attach_edge(
        structure.copy(),
        colors.copy(),
        element_edge_min_ratio=0.2,
        element_edge_max_ratio=0.5,
        step=2,
        colors_dict=colors_dict,
        verbose=True,
    )

    log_info(f"Edge test complete. Filled voxels: {np.sum(structure > 0)}")

    # Test plane attachment
    structure, colors = attach_plane(
        structure.copy(),
        colors.copy(),
        element_plane_min_ratio=0.3,
        element_plane_max_ratio=0.6,
        step=2,
        colors_dict=colors_dict,
        verbose=True,
    )

    log_info(f"Plane test complete. Filled voxels: {np.sum(structure > 0)}")

    # Test pipe attachment
    structure, colors = attach_pipe(
        structure.copy(),
        colors.copy(),
        element_volume_min_ratio=0.4,
        element_volume_max_ratio=0.7,
        step=2,
        colors_dict=colors_dict,
        verbose=True,
    )

    log_info(f"Pipe test complete. Filled voxels: {np.sum(structure > 0)}")

    # Test grid attachment
    structure, colors = attach_grid(
        structure.copy(), colors.copy(), step=4, colors_dict=colors_dict, verbose=True
    )

    log_info(f"Grid test complete. Filled voxels: {np.sum(structure > 0)}")

    # Test generic shape attachment
    structure, colors = attach_shape(
        structure.copy(),
        colors.copy(),
        ShapeType.EDGE,
        min_ratio=0.2,
        max_ratio=0.5,
        step=2,
        colors_dict=colors_dict,
        verbose=True,
    )

    log_info(f"Generic shape test complete. Filled voxels: {np.sum(structure > 0)}")
