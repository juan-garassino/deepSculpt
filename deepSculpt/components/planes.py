import random
import numpy as np

from deepSculpt.components.utils import return_axis
from deepSculpt.params import COLOR_PLANES


def add_plane(
    void, color_void, element_plane_min, element_plane_max, step, verbose
):  # WHAT TO DO WITH THE WORKING PLANE PARAMETER

    element = None
    delta = None
    top_left_corner = None
    bottom_right_corner = None
    working_plane = return_axis(void, color_void)[0]
    color_parameters = return_axis(void, color_void)[1]

    # section = None

    if verbose == True:
        print(working_plane)
        print("###############################################################")

    # Variables
    element = np.ones(
        (
            random.randrange(element_plane_min, element_plane_max, step),
            random.randrange(element_plane_min, element_plane_max, step),
        )
    )
    # creates the element to be inserted
    delta = np.array(working_plane.shape) - np.array(element.shape)
    # finds the delta between the size of the void and the size of the element
    top_left_corner = np.array(
        (
            np.random.randint(low=0, high=delta[0]),
            np.random.randint(low=0, high=delta[1]),
        )
    )
    # finds the coordinates of the top left corner
    bottom_right_corner = np.array(top_left_corner) + np.array(
        element.shape
    )  # - np.array([1,1]))
    # finds the coordinates of the bottom right corner
    working_plane[
        top_left_corner[0] : bottom_right_corner[0],
        top_left_corner[1] : bottom_right_corner[1],
    ] = element
    # makes the slides using the coordinates equal to the element

    color_parameters[
        top_left_corner[0] : bottom_right_corner[0],
        top_left_corner[1] : bottom_right_corner[1],
    ] = COLOR_PLANES

    if verbose == True:
        print_information()
        print("###############################################################")

    return void.astype("int8"), color_void
