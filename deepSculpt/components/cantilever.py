import random
import numpy as np

from deepSculpt.params import COLOR_VOLUMES


def add_pipe_cantilever(
    void, color_void, element_volume_min, element_volume_max, step, verbose
):  # THIS IS GOOD!!

    element = None
    working_plane = None
    delta = None
    top_left_corner = None
    bottom_right_corner = None
    axis_selection = np.random.randint(low=0, high=2)
    shape_selection = np.random.randint(low=0, high=2)
    depth = random.randrange(element_volume_min, element_volume_max, step)

    if verbose == True:
        print(working_plane)
        print("###############################################################")

    element = np.ones(
        (
            random.randrange(element_volume_min, element_volume_max, step),
            random.randrange(element_volume_min, element_volume_max, step),
        )
    )
    element = np.repeat(element, repeats=depth, axis=0).reshape(
        (element.shape[0], element.shape[1], depth)
    )

    element_void = np.zeros((element.shape[0] - 2, element.shape[1] - 2))
    element_void = np.repeat(element_void, repeats=depth).reshape(
        (element_void.shape[0], element_void.shape[1], depth)
    )

    # element[1:-1,1:-1,:] = element_void # elegir pasar el vacio o no como parte del volumen

    delta = np.array(void.shape) - np.array(
        element.shape
    )  # ENCONTRAR LOS NUEVOS DELTAS

    corner_1 = np.array(
        (
            np.random.randint(low=0, high=delta[0]),
            np.random.randint(low=0, high=delta[1]),
            np.random.randint(low=0, high=delta[2]),
        )
    )
    corner_2 = np.array((corner_1[0] + element.shape[0], corner_1[1], corner_1[2]))
    corner_3 = np.array((corner_1[0], corner_1[1], corner_1[2] + element.shape[2]))
    corner_4 = np.array(
        (
            corner_1[0] + element.shape[0],
            corner_1[1],
            corner_1[2] + element.shape[2],
        )
    )

    corner_5 = np.array((corner_1[0], corner_1[1] + element.shape[1], corner_1[2]))
    corner_6 = np.array((corner_2[0], corner_2[1] + element.shape[1], corner_2[2]))
    corner_7 = np.array((corner_3[0], corner_3[1] + element.shape[1], corner_3[2]))
    corner_8 = np.array((corner_4[0], corner_4[1] + element.shape[1], corner_4[2]))

    color_volume = np.random.randint(0, len(COLOR_VOLUMES))

    # creates the floor and ceiling
    void[
        corner_3[0] : corner_8[0], corner_3[1] : corner_8[1], corner_3[2] - 1
    ] = element[:, :, 0]
    color_void[
        corner_3[0] : corner_8[0], corner_3[1] : corner_8[1], corner_3[2] - 1
    ] = COLOR_VOLUMES[color_volume]

    void[corner_1[0] : corner_6[0], corner_1[1] : corner_6[1], corner_1[2]] = element[
        :, :, 1
    ]
    color_void[
        corner_1[0] : corner_6[0], corner_1[1] : corner_6[1], corner_1[2]
    ] = COLOR_VOLUMES[color_volume]

    # creates de walls
    if shape_selection == 0:
        if axis_selection == 0:
            void[
                corner_1[0], corner_1[1] : corner_7[1], corner_1[2] : corner_7[2]
            ] = element[0, :, :]
            color_void[
                corner_1[0], corner_1[1] : corner_7[1], corner_1[2] : corner_7[2]
            ] = COLOR_VOLUMES[color_volume]

            void[
                corner_2[0] - 1,
                corner_2[1] : corner_8[1],
                corner_2[2] : corner_8[2],
            ] = element[1, :, :]
            color_void[
                corner_2[0] - 1,
                corner_2[1] : corner_8[1],
                corner_2[2] : corner_8[2],
            ] = COLOR_VOLUMES[color_volume]
        else:
            void[
                corner_5[0] : corner_8[0], corner_5[1], corner_5[2] : corner_8[2]
            ] = element[:, 0, :]
            color_void[
                corner_5[0] : corner_8[0], corner_5[1], corner_5[2] : corner_8[2]
            ] = COLOR_VOLUMES[color_volume]

            void[
                corner_1[0] : corner_4[0], corner_1[1], corner_1[2] : corner_4[2]
            ] = element[:, 0, :]
            color_void[
                corner_1[0] : corner_4[0], corner_1[1], corner_1[2] : corner_4[2]
            ] = COLOR_VOLUMES[color_volume]

    else:
        if axis_selection == 0:
            void[
                corner_1[0], corner_1[1] : corner_7[1], corner_1[2] : corner_7[2]
            ] = element[0, :, :]
            color_void[
                corner_1[0], corner_1[1] : corner_7[1], corner_1[2] : corner_7[2]
            ] = COLOR_VOLUMES[color_volume]

            void[
                corner_5[0] : corner_8[0], corner_5[1], corner_5[2] : corner_8[2]
            ] = element[:, 0, :]
            color_void[
                corner_5[0] : corner_8[0], corner_5[1], corner_5[2] : corner_8[2]
            ] = COLOR_VOLUMES[color_volume]
        else:
            void[
                corner_2[0] - 1,
                corner_2[1] : corner_8[1],
                corner_2[2] : corner_8[2],
            ] = element[1, :, :]
            color_void[
                corner_2[0] - 1,
                corner_2[1] : corner_8[1],
                corner_2[2] : corner_8[2],
            ] = COLOR_VOLUMES[color_volume]

            void[
                corner_1[0] : corner_4[0], corner_1[1], corner_1[2] : corner_4[2]
            ] = element[:, 0, :]
            color_void[
                corner_1[0] : corner_4[0], corner_1[1], corner_1[2] : corner_4[2]
            ] = COLOR_VOLUMES[color_volume]

    if verbose == True:
        print_information()
        print("###############################################################")

    return void, color_void
