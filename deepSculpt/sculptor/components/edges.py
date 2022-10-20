import random
import numpy as np

from deepSculpt.sculptor.components.utils import return_axis
from deepSculpt.curator.tools.params import COLOR_EDGES


def add_edge(
    void, color_void, element_edge_min, element_edge_max, step, verbose
):  # WHAT TO DO WITH THE WORKING PLANE PARAMETER

    working_plane = return_axis(void, color_void)[0]
    color_parameters = return_axis(void, color_void)[1]
    # selection of the axis to work on

    if verbose == True:
        print(working_plane)
        print("###############################################################")

    # Variables
    edge_length = random.randrange(
        element_edge_min, element_edge_max, step
    )  # estas variables quizas no necesiten ser self!!
    edge_plane = np.random.randint(low=0, high=2)

    if edge_plane == 0:
        element = np.ones(edge_length).reshape(edge_length, 1)
    else:
        element = np.ones(edge_length).reshape(edge_length, 1).T

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
    ] = COLOR_EDGES

    if verbose == True:
        print(working_plane)
        print("###############################################################")

    return void.astype("int8"), color_void
