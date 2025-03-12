import numpy as np
import random
from enum import Enum
from typing import List, Tuple
import os

class ShapeType(Enum):
    EDGE = 1
    PLANE = 2
    PIPE = 3

def generate_random_size(min_ratio: float, max_ratio: float, base_size: int, dimensions: int) -> List[int]:
    """
    Generate random sizes for a shape based on given ratios and dimensions.
    
    Input:
    - min_ratio: Minimum size ratio relative to base_size
    - max_ratio: Maximum size ratio relative to base_size
    - base_size: Reference size (usually the smallest dimension of the void)
    - dimensions: Number of dimensions for the shape (1 for edge, 2 for plane, 3 for pipe)
    
    Output:
    - List of integers representing the size in each dimension
    
    Used for: Determining the size of the shape to be inserted
    """
    return [max(int(random.uniform(min_ratio, max_ratio) * base_size), 2) for _ in range(dimensions)]

def select_random_position(void_shape: Tuple[int, ...], shape_size: List[int]) -> List[int]:
    """
    Select a random position to insert a shape in the void.
    
    Input:
    - void_shape: Dimensions of the void (3D tuple)
    - shape_size: Dimensions of the shape to be inserted
    
    Output:
    - List of integers representing the insertion position
    
    Used for: Determining where to place the shape in the void
    """
    return [random.randint(0, max(0, void_dim - shape_dim)) for void_dim, shape_dim in zip(void_shape, shape_size)]

def create_shape(shape_type: ShapeType, shape_size: List[int]) -> np.ndarray:
    """
    Create a shape (edge, plane, or pipe) of ones with given dimensions and size.
    
    Input:
    - shape_type: Enum indicating the type of shape (EDGE, PLANE, or PIPE)
    - shape_size: Dimensions of the shape
    
    Output:
    - NumPy array representing the shape (1s where the shape is, 0s elsewhere)
    
    Used for: Generating the shape to be inserted into the void
    """
    if shape_type == ShapeType.PIPE:
        # Create a hollow pipe
        shape = np.zeros(shape_size)
        shape[0, :, :] = shape[-1, :, :] = shape[:, 0, :] = shape[:, -1, :] = shape[:, :, 0] = shape[:, :, -1] = 1
    else:
        shape = np.ones(shape_size)
    return shape

def insert_shape(void: np.ndarray, shape: np.ndarray, position: List[int]) -> np.ndarray:
    """
    Insert a shape into the void at the given position.
    
    Input:
    - void: 3D NumPy array representing the space
    - shape: NumPy array representing the shape to insert
    - position: List of coordinates where to insert the shape
    
    Output:
    - Updated void array with the shape inserted
    
    Used for: Adding the shape to the void space
    """
    slices = tuple(slice(pos, pos + dim) for pos, dim in zip(position, shape.shape))
    void[slices] = shape
    return void

def assign_color(color_void: np.ndarray, shape: np.ndarray, position: List[int], color: str) -> np.ndarray:
    """
    Assign color to the shape in the color void array.
    
    Input:
    - color_void: 3D NumPy array of objects representing colors
    - shape: NumPy array representing the shape
    - position: List of coordinates where the shape is inserted
    - color: Color to assign to the shape
    
    Output:
    - Updated color_void array with colors assigned to the shape
    
    Used for: Coloring the inserted shape in the color representation of the void
    """
    slices = tuple(slice(pos, pos + dim) for pos, dim in zip(position, shape.shape))
    color_void[slices] = np.where(shape == 1, color, color_void[slices])
    return color_void

def select_axes(num_axes: int) -> List[int]:
    """
    Select random axes for shape placement.
    
    Input:
    - num_axes: Number of axes to select (1 for edge, 2 for plane, 3 for pipe)
    
    Output:
    - List of selected axes indices
    
    Used for: Determining which axes the shape will span in the void
    """
    return random.sample(range(3), num_axes)

def print_verbose(message: str) -> None:
    """
    Print verbose output if enabled.
    
    Input:
    - message: String message to print
    
    Output:
    - None (prints to console)
    
    Used for: Debugging and providing detailed information about the shape insertion process
    """
    if int(os.environ.get("VERBOSE", 0)) == 1:
        print(f"\n ⏹ {message}")

def validate_dimensions(shape_size: List[int], void_shape: Tuple[int, ...]) -> bool:
    """
    Validate that the shape fits within the void dimensions.
    
    Input:
    - shape_size: Dimensions of the shape
    - void_shape: Dimensions of the void
    
    Output:
    - Boolean indicating if the shape fits in the void
    
    Used for: Ensuring the generated shape doesn't exceed the void's dimensions
    """
    return all(s <= v for s, v in zip(shape_size, void_shape))

def select_shape_type() -> ShapeType:
    """
    Randomly select a shape type (edge, plane, or pipe).
    
    Input:
    - None
    
    Output:
    - Randomly selected ShapeType enum
    
    Used for: Choosing which type of shape to create when generating random shapes
    """
    return random.choice(list(ShapeType))

def attach_shape(
    void: np.ndarray,
    color_void: np.ndarray,
    shape_type: ShapeType,
    min_ratio: float,
    max_ratio: float,
    colors: dict
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Main function to attach a shape of specified type to the void.
    
    Input:
    - void: 3D NumPy array representing the space
    - color_void: 3D NumPy array of objects representing colors
    - shape_type: Type of shape to attach (EDGE, PLANE, or PIPE)
    - min_ratio: Minimum size ratio for the shape
    - max_ratio: Maximum size ratio for the shape
    - colors: Dictionary of colors for each shape type
    
    Output:
    - Tuple of updated void and color_void arrays
    
    Used for: Orchestrating the entire process of creating and attaching a shape to the void
    """
    axes = select_axes(shape_type.value)
    shape_size = generate_random_size(min_ratio, max_ratio, min(void.shape), shape_type.value)
    full_shape_size = [1] * 3
    for axis, size in zip(axes, shape_size):
        full_shape_size[axis] = size
    
    if not validate_dimensions(full_shape_size, void.shape):
        print_verbose(f"Shape size {full_shape_size} exceeds void dimensions {void.shape}. Skipping.")
        return void, color_void
    
    shape = create_shape(shape_type, full_shape_size)
    position = select_random_position(void.shape, full_shape_size)
    
    void = insert_shape(void, shape, position)
    color = colors[shape_type.name.lower() + 's']
    if isinstance(color, list):
        color = random.choice(color)
    color_void = assign_color(color_void, shape, position, color)
    
    print_verbose(f"Attached {shape_type.name} at position {position} with size {full_shape_size}")
    
    return void, color_void

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_voxels(void: np.ndarray, color_void: np.ndarray, filename: str = None):
    """
    Plot the 3D void using voxels.
    
    Input:
    - void: 3D NumPy array representing the space
    - color_void: 3D NumPy array of objects representing colors
    - filename: If provided, save the plot to this file instead of displaying
    
    Output:
    - None (displays or saves the plot)
    
    Used for: Visualizing the 3D void with inserted shapes as voxels
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a boolean array where True represents filled voxels
    filled = void > 0
    
    # Create color array
    colors = np.empty(void.shape, dtype=object)
    colors[filled] = color_void[filled]
    
    # Plot voxels
    ax.voxels(filled, facecolors=colors, edgecolor='k')
    
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('3D Visualization of the Void')
    
    if filename:
        plt.savefig(filename)
        plt.close(fig)
    else:
        plt.show()

# Usage example:
if __name__ == "__main__":
    void = np.zeros((20, 20, 20))
    color_void = np.empty(void.shape, dtype=object)
    colors = {
        'edges': 'red',
        'planes': 'green',
        'pipes': ['blue', 'cyan', 'magenta']
    }

    for i in range(10):  # Attach 10 random shapes
        shape_type = select_shape_type()
        void, color_void = attach_shape(void, color_void, shape_type, 0.1, 0.9, colors)
        
        # Save an intermediate plot after each shape attachment
        plot_3d_voxels(void, color_void, f'void_state_{i+1}.png')

    # Display the final result
    plot_3d_voxels(void, color_void)