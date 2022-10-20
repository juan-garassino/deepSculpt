import numpy as np
import random
from deepSculpt.manager.components.utils import return_axis
from deepSculpt.tools.params import COLOR_EDGES


def add_grid(void, color_void, element_grid_min, element_grid_max, step, verbose):
    section = return_axis(void, color_void)[2]
    working_plane = void[:, :, section]
    color_parameters = return_axis(void, color_void)[1]
    # selection of the axis to work on

    if verbose == True:
        print(working_plane)
        print("###############################################################")

    column_height = random.randrange(element_grid_min, element_grid_max, step)
    element = np.ones(column_height).reshape(column_height, 1)

    x = np.arange(start=1, stop=49, step=9)
    y = np.arange(start=1, stop=49, step=9)

    z_base = np.zeros((6,))
    z_top = z_base + column_height

    grid_coor_base = np.array(np.meshgrid(x, y, z_base)).T[0]
    grid_coor_top = np.array(np.meshgrid(x, y, z_top)).T[0]
    grid_coor = np.concatenate((grid_coor_base, grid_coor_top), axis=2).astype(int)

    for column_row in grid_coor:
        for column in column_row:
            void[
                column[0] : column[3] + 1,
                column[1] : column[4] + 1,
                column[2] : column[5],
            ] = element.reshape((column_height,))

    for column_row in grid_coor:
        for column in column_row:
            color_void[
                column[0] : column[3] + 1,
                column[1] : column[4] + 1,
                column[2] : column[5],
            ] = COLOR_EDGES

    if verbose == True:
        print(working_plane)
        print("###############################################################")

    return void.astype("int8"), color_void
