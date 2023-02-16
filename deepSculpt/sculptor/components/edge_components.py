from deepSculpt.manager.manager import Manager
from deepSculpt.manager.tools.params import COLORS

import random
import numpy as np
from colorama import Fore, Style


def attach_edge(
    void: np.ndarray,
    color_void: np.ndarray,
    element_edge_min_ratio: float = 0.1,
    element_edge_max_ratio: float = 0.9,
    step: int = 4,
    verbose: bool = False,
):  # -> tuple[np.ndarray, np.ndarray]:
    """
    This function adds an edge of random length to the input array.
    Args:
        void (numpy array): Input array to add the edge to.
        color_void (numpy array): Array containing the color information of the input array.
        element_edge_min_ratio (float): The minimum length of the edge as a ratio of the input array size.
        element_edge_max_ratio (float): The maximum length of the edge as a ratio of the input array size.
        step (int): The step size of the edge length.
        verbose (bool): Verbosity flag for printing additional information.

    Returns:
        void (numpy array): The input array with an edge added to it.
        color_void (numpy array): The input color array with the corresponding color information of the added edge.

    """
    # Calculate the minimum and maximum length of the edge
    element_edge_min_index = int(element_edge_min_ratio * void.shape[0])
    element_edge_max = int(element_edge_max_ratio * void.shape[0])

    # Select the axis to work on
    working_plane, color_parameters, section = Manager.return_axis(void, color_void)

    # Print the working axis and the input array
    if verbose == True:
        print(
            "\n ⏹ "
            + Fore.RED
            + f"Working on axis shaped {working_plane.shape}"
            + Style.RESET_ALL
        )
        print("\n ⏹ " + Fore.RED + f"Input array shaped {void.shape}" + Style.RESET_ALL)
        print("\n ⏹ " + Fore.RED + "----------------------------" + Style.RESET_ALL)

    # Generate the edge length and orientation
    edge_length = random.randrange(element_edge_min_index, element_edge_max, step)
    edge_plane = np.random.randint(low=0, high=2)

    # Create the edge element to be inserted
    if edge_plane == 0:
        element = np.ones(edge_length).reshape(edge_length, 1)
    else:
        element = np.ones(edge_length).reshape(edge_length, 1).T

    # Find the delta between the size of the input array and the size of the edge element
    delta = np.array(working_plane.shape) - np.array(element.shape)

    # Find the coordinates of the top left corner of the element
    top_left_corner = np.array(
        (
            np.random.randint(low=0, high=delta[0]),
            np.random.randint(low=0, high=delta[1]),
        )
    )

    # Find the coordinates of the bottom right corner of the element
    bottom_right_corner = np.array(top_left_corner) + np.array(element.shape)

    # Insert the element into the input array
    working_plane[
        top_left_corner[0] : bottom_right_corner[0],
        top_left_corner[1] : bottom_right_corner[1],
    ] = element

    # Update the color array with the color information of the added edge
    color_parameters[
        top_left_corner[0] : bottom_right_corner[0],
        top_left_corner[1] : bottom_right_corner[1],
    ] = COLORS["edges"]

    # Print the updated array
    if verbose:
        print(
            "\n ⏹ " + Fore.RED + f"Output array shaped {void.shape}" + Style.RESET_ALL
        )
        print("\n ⏹ " + Fore.RED + "----------------------------" + Style.RESET_ALL)

    # Return the input array and the corresponding color array
    return void.astype("int8"), color_void


if __name__ == "__main__":
    # Create an initial void and color void
    volumes_void = np.zeros((10, 10, 10))

    materials_void = np.empty(volumes_void.shape, dtype=object)

    # Add an edge to the void
    void, color_void = attach_edge(
        volumes_void, materials_void, 0.1, 0.9, 3, verbose=True
    )

    print(void)

    print(color_void)
