from deepSculpt.curator.tools.params import COLOR_EDGES
from colorama import Fore, Style

import random
import numpy as np


def add_grid(volumes_void=None, materials_void=None, step=1, verbose=False):

    void_dim = volumes_void.shape[0]

    if verbose == True:
        print(
            "\n⏹ "
            + Fore.RED
            + f"The color of the grid is {COLOR_EDGES}"
            + Style.RESET_ALL
        )

    locations = []

    if void_dim % 2 == 0:

        left_position = void_dim / 2 - (step / 2)

        right_position = left_position + (step + 1)

        locations.append(left_position)

        locations.append(right_position)

        while left_position > 0 and right_position < (void_dim - (step + 1)):

            left_position = left_position - (step + 1)

            right_position = right_position + (step + 1)

            locations.append(left_position)

            locations.append(right_position)

    if void_dim % 2 != 0:
        pass

    X = np.array(sorted(locations)) - 1

    Y = np.array(sorted(locations)) - 1

    Z = np.array(random.choices(locations[1:], k=len(X) * len(Y))).reshape(
        (len(X), len(Y))
    )

    bases = np.zeros(
        (
            len(X),
            len(Y),
        )
    ).reshape((len(X), len(Y), 1))

    heights = Z.reshape((len(X), len(Y), 1))  # random heights selected

    grid = np.array(np.meshgrid(X, Y)).T  # X Y grid created

    column_top_coordinates = (
        np.concatenate((grid, heights), axis=2)
        .reshape((len(X) * len(Y), 3))
        .astype("int8")
    )

    column_base_coordinates = (
        np.concatenate((grid, bases), axis=2)
        .reshape((len(X) * len(Y), 3))
        .astype("int8")
    )

    if verbose == True:
        pass
        # print(column_top_coordinates)

        # print(column_base_coordinates)

    for column_base_coordinate, column_top_coordinate in zip(
        list(column_base_coordinates), list(column_top_coordinates)
    ):
        volumes_void[
            column_base_coordinate[0],
            column_base_coordinate[1],
            column_base_coordinate[2] : column_top_coordinate[2],
        ] = 1

    volumes_void[:, :, 0] = 1

    materials_void[volumes_void == 1] = COLOR_EDGES # 1

    return volumes_void, materials_void # np.where(materials_void == 1, COLOR_EDGES, 0)
