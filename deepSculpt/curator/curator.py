from deepSculpt.sculptor.sculptor import Sculptor
from deepSculpt.manager.manager import Manager
from deepSculpt.curator.tools.params import (
    N_EDGE_ELEMENTS,
    N_PLANE_ELEMENTS,
    N_VOLUME_ELEMENTS,
    COLOR_EDGES,
    COLOR_PLANES,
    COLOR_VOLUMES,
    ELEMENT_EDGE_MIN,
    ELEMENT_EDGE_MAX,
    ELEMENT_PLANE_MIN,
    ELEMENT_PLANE_MAX,
    ELEMENT_VOLUME_MIN,
    ELEMENT_VOLUME_MAX,
    VERBOSE,
)

from datetime import date
import numpy as np
import os
import time
from colorama import Fore, Style


class Curator:
    def __init__(self, create=False, locally=True, path_volumes="", path_colors=""):
        self.locally = locally
        self.create = create
        self.path_volumes = path_volumes
        self.path_colors = path_colors

    def create_sculpts(
        self,
        directory,
        n_samples=int(os.environ.get("N_SAMPLES_CREATE")),
        n_edge_elements=N_EDGE_ELEMENTS,
        n_plane_elements=N_PLANE_ELEMENTS,
        n_volume_elements=N_VOLUME_ELEMENTS,
        color_edges=COLOR_EDGES,
        color_planes=COLOR_PLANES,
        color_volumes=COLOR_VOLUMES,
        verbose=VERBOSE,
        void_dim=int(os.environ.get("VOID_DIM")),
    ):

        raw_data = []

        color_raw_data = []

        count = 0

        for count, sculpture in enumerate(range(n_samples)):  #

            print(
                "\n⏹ "
                + Fore.BLUE
                + f"Creating sculpture number {count}"
                + Style.RESET_ALL
            )

            start = time.time()

            if (count + 1) % 25 == 0:
                print(
                    "\n⏹ "
                    + Fore.GREEN
                    + "{} sculputers where created in {}".format(
                        (count + 1), time.time() - start
                    )
                    + Style.RESET_ALL
                )

                # print("\r{0}".format(count), end="")

            sculptor = Sculptor(
                void_dim=void_dim,
                edges=(
                    n_edge_elements,
                    ELEMENT_EDGE_MIN,
                    ELEMENT_EDGE_MAX,
                ),  # number of elements, minimun, maximun
                planes=(n_plane_elements, ELEMENT_PLANE_MIN, ELEMENT_PLANE_MAX),
                volumes=(n_volume_elements, ELEMENT_VOLUME_MIN, ELEMENT_VOLUME_MAX),
                # grid=(2, 5), # minimun height of column, and maximun height
                materials_edges=COLOR_EDGES,
                materials_planes=COLOR_PLANES,
                materials_volumes=COLOR_VOLUMES,
                step=int(void_dim / 6),
                verbose=verbose,
            )

            sculpture = sculptor.generative_sculpt()

            raw_data.append(
                sculpture[0].astype("int8")
            )  # NOT APPEND BUT SAVE IN DIFF FILES!!

            color_raw_data.append(sculpture[1])

        raw_data = (
            np.asarray(raw_data)
            .reshape(
                (
                    int(os.environ.get("N_SAMPLES_CREATE")),
                    int(os.environ.get("VOID_DIM")),
                    int(os.environ.get("VOID_DIM")),
                    int(os.environ.get("VOID_DIM")),
                )
            )
            .astype("int8")
        )

        color_raw_data = (
            np.asarray(color_raw_data)
            .reshape(
                (
                    int(os.environ.get("N_SAMPLES_CREATE")),
                    int(os.environ.get("VOID_DIM")),
                    int(os.environ.get("VOID_DIM")),
                    int(os.environ.get("VOID_DIM")),
                )
            )
            .astype("object")
        )

        Manager.make_directory(directory)

        np.save(
            f"{directory}/sample-volumes[{date.today()}]", raw_data, allow_pickle=True
        )

        np.save(
            f"{directory}/sample-colors[{date.today()}]",
            color_raw_data,
            allow_pickle=True,
        )

        print(
            "\n🔽 "
            + Fore.BLUE
            + f"Just created 'raw_data' shaped {raw_data.shape} and 'color_raw_data' shaped{color_raw_data.shape}"
            + Style.RESET_ALL
        )

        return (raw_data, color_raw_data)


if __name__ == "__main__":
    curator = Curator()

    out_dir = os.path.join(
        os.environ.get("HOME"), "code", "juan-garassino", "deepSculpt", "data"
    )

    curator.create_sculpts(out_dir)
