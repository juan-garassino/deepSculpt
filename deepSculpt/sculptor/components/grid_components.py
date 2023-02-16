from deepSculpt.manager.manager import Manager
from deepSculpt.manager.tools.params import COLORS

import random
import numpy as np
from colorama import Fore, Style


def attach_grid(volumes_void, materials_void, step=1, verbose=False):
    """
    Adds a grid structure to a given 3D volume.

    Args:
    volumes_void: a 3D numpy array of shape (height, width, depth)
    materials_void: a 3D numpy array of the same shape as volumes_void
    step (int): the grid step size
    verbose (bool): if True, prints progress messages

    Returns:
    A tuple of two 3D numpy arrays:
    1. The modified volumes_void array with the grid structure added
    2. The modified materials_void array with the grid structure colored
    in the COLOR_EDGES value defined in params.py
    """
    void_dim = volumes_void.shape[0]

    if verbose == True:
        print(
            "\n ⏹ "
            + Fore.RED
            + f"The color of the grid is {COLORS['edges']}"
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

    Z = np.array(random.choices(locations[2:], k=len(X) * len(Y))).reshape(
        (len(X), len(Y))
    )

    bases = np.zeros((len(X), len(Y), 1)).reshape((len(X), len(Y), 1))
    heights = Z.reshape((len(X), len(Y), 1))
    grid = np.array(np.meshgrid(X, Y)).T
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

    if verbose:
        Manager.verbose(bases, heights, column_base_coordinates)

    for column_base_coordinate, column_top_coordinate in zip(
        list(column_base_coordinates), list(column_top_coordinates)
    ):
        volumes_void[
            column_base_coordinate[0],
            column_base_coordinate[1],
            column_base_coordinate[2] : column_top_coordinate[2],
        ] = 1

    volumes_void[:, :, 0] = 1
    materials_void[volumes_void == 1] = COLORS["edges"]

    return volumes_void.astype("int8"), materials_void


if __name__ == "__main__":
    # Example usage
    volumes_void = np.zeros((50, 50, 50))
    materials_void = np.empty(volumes_void.shape, dtype=object)
    volumes_void, materials_void = attach_grid(
        volumes_void=volumes_void, materials_void=materials_void, step=5, verbose=True
    )
    print(volumes_void)
    print(materials_void)
