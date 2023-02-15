from deepSculpt.manager.manager import Manager
from deepSculpt.curator.tools.params import COLORS

import random
import numpy as np
from colorama import Fore, Style
import os

def attach_plane(
    void: np.ndarray,
    color_void: np.ndarray,
    element_plane_min_ratio: float = 0.1,
    element_plane_max_ratio: float = 0.9,
    step: int = 4,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Adds a plane of ones to a given 2D numpy array.

    Args:
        void: The 2D numpy array to which the plane is added.
        color_void: The corresponding 2D color numpy array.
        element_plane_min_ratio: The minimum size of the plane as a ratio of the size of the void.
        element_plane_max_ratio: The maximum size of the plane as a ratio of the size of the void.
        step: The step size between elements in the plane.
        verbose: Whether to print information about the operation.

    Returns:
        A tuple of two 2D numpy arrays. The first array is the modified void, and the second is the corresponding
        color numpy array.
    """

    # Calculate the indices of the start and end of the plane.
    element_plane_min_index = int(element_plane_min_ratio * void.shape[0])
    element_plane_max = int(element_plane_max_ratio * void.shape[0])

    # Initialize variables.
    element, delta, top_left_corner, bottom_right_corner = (None, None, None, None)

    # Get the working plane and color parameters.
    working_plane, color_parameters, section = Manager.return_axis(
        void, color_void)

    if verbose == True:
        print("\n ⏹ " + Fore.RED +
              f"The color of the plane is {COLORS['planes']}" + Style.RESET_ALL)

    # Create the element to be inserted.
    element = np.ones(
        (
            random.randrange(element_plane_min_index, element_plane_max, step),
            random.randrange(element_plane_min_index, element_plane_max, step),
        )
    )

    # Find the delta between the size of the void and the size of the element.
    delta = np.array(working_plane.shape) - np.array(element.shape)

    # Find the coordinates of the top left corner.
    top_left_corner = np.array(
        (
            np.random.randint(low=0, high=delta[0]),
            np.random.randint(low=0, high=delta[1]),
        )
    )

    # Find the coordinates of the bottom right corner.
    bottom_right_corner = np.array(top_left_corner) + np.array(element.shape)

    # Add the element to the working plane.
    working_plane[
        top_left_corner[0] : bottom_right_corner[0],
        top_left_corner[1] : bottom_right_corner[1],
    ] = element

    # Set the color of the added plane.
    color_parameters[
        top_left_corner[0]:bottom_right_corner[0],
        top_left_corner[1]:bottom_right_corner[1], ] = COLORS['planes']

    if os.environ.get("VERBOSE") == 1:
        Manager.verbose(
            void=void,
            element=element,
            delta=delta,
            top_left_corner=top_left_corner,
            bottom_right_corner=bottom_right_corner,
        )

    return void.astype("int8"), color_void


if __name__ == "__main__":
    # Create an initial void and color void
    volumes_void = np.zeros((10, 10, 10))

    materials_void = np.empty(volumes_void.shape, dtype=object)

    # Add an edge to the void
    volumes_void, materials_void = attach_plane(
        volumes_void, materials_void, 0.1, 0.9, 3, verbose=True
    )

    print(volumes_void)

    print(materials_void)
