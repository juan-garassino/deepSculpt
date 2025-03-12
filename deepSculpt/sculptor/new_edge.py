from deepSculpt.manager.manager import Manager
from deepSculpt.manager.tools.params import COLORS

import random
import numpy as np
from colorama import Fore, Style
from functools import wraps
import matplotlib.pyplot as plt

def random_edge_size(min_ratio, max_ratio, base_size):
    min_size = max(int(min_ratio * base_size), 2)  # Ensure minimum size of 2
    max_size = int(max_ratio * base_size)
    return random.randint(min_size, max_size)

def random_edge_position(max_position, edge_size):
    return random.randint(0, max_position - edge_size)

def print_verbose(message):
    print("\n ⏹ " + Fore.RED + message + Style.RESET_ALL)

def verbose_output(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        verbose = kwargs.get('verbose', False)
        if verbose:
            print_verbose(f"Starting {func.__name__}")
        result = func(*args, **kwargs)
        if verbose:
            print_verbose(f"Finished {func.__name__}")
        return result
    return wrapper

@verbose_output
def attach_edge(
    void: np.ndarray,
    color_void: np.ndarray,
    element_edge_min_ratio: float = 0.1,
    element_edge_max_ratio: float = 0.9,
    verbose: bool = False,
):  # -> tuple[np.ndarray, np.ndarray]:
    """
    This function adds an edge of random length to the input array.
    Args:
        void (numpy array): Input array to add the edge to.
        color_void (numpy array): Array containing the color information of the input array.
        element_edge_min_ratio (float): The minimum length of the edge as a ratio of the input array size.
        element_edge_max_ratio (float): The maximum length of the edge as a ratio of the input array size.
        verbose (bool): Verbosity flag for printing additional information.

    Returns:
        void (numpy array): The input array with an edge added to it.
        color_void (numpy array): The input color array with the corresponding color information of the added edge.
    """
    axis = random.randint(0, 2)  # Randomly choose axis (0, 1, or 2)

    if verbose:
        print_verbose(f"Working on axis {axis}")
        print_verbose(f"Input array shaped {void.shape}")

    edge_size = random_edge_size(element_edge_min_ratio, element_edge_max_ratio, void.shape[axis])
    edge_start = random_edge_position(void.shape[axis], edge_size)

    if verbose:
        print_verbose(f"Edge size: {edge_size}")
        print_verbose(f"Edge start position: {edge_start}")

    # Create the position tuple for the edge
    position = [0, 0, 0]
    position[axis] = slice(edge_start, edge_start + edge_size)
    for i in range(3):
        if i != axis:
            position[i] = random.randint(0, void.shape[i] - 1)

    if verbose:
        print_verbose(f"Edge position: {position}")

    # Insert the edge into the void array
    void[tuple(position)] = 1

    # Update the color array
    color_void[tuple(position)] = COLORS["edges"]

    return void.astype("int8"), color_void

def plot_3d_array(arr, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    x, y, z = np.where(arr == 1)
    ax.scatter(x, y, z, c='r', marker='o')

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('3D Visualization of the Array')

    plt.savefig(filename)
    plt.close(fig)  # Close the figure to free up memory

if __name__ == "__main__":
    for i in range(10):
        volumes_void = np.zeros((10, 10, 10))
        materials_void = np.empty(volumes_void.shape, dtype=object)

        void, color_void = attach_edge(
            volumes_void, materials_void, 0.1, 0.9, verbose=True
        )

        print(f"\nIteration {i+1}:")
        print("Resulting void array:")
        print(void)
        print("\nResulting color array:")
        print(color_void)

        # Plot and save the 3D array
        plot_3d_array(void, f'edge_plot_{i+1}.png')

    print("\nAll 10 plots have been saved.")