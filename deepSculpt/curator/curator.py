from deepSculpt.sculptor.sculptor import Sculptor
from deepSculpt.manager.manager import Manager
from deepSculpt.curator.tools.params import (
    #    N_EDGE_ELEMENTS,
    #    N_PLANE_ELEMENTS,
    #    N_VOLUME_ELEMENTS,
    COLOR_EDGES,
    COLOR_PLANES,
    COLOR_VOLUMES,
    #    ELEMENT_EDGE_MIN,
    #    ELEMENT_EDGE_MAX,
    #    ELEMENT_PLANE_MIN,
    #    ELEMENT_PLANE_MAX,
    #    ELEMENT_VOLUME_MIN,
    #    ELEMENT_VOLUME_MAX,
    #    VERBOSE,
)

from datetime import date
import numpy as np
import os
import time
from colorama import Fore, Style


class Curator:
    def __init__(
        self,
        void_dim=32,  # locally=True, path_volumes="", path_colors="",
        edge_elements=(0, 0.3, 0.5),
        plane_elements=(0, 0.3, 0.5),
        volume_elements=(0, 0.3, 0.5),
        step=None,
        directory=None,
        n_samples=100,
        grid=1,
    ):

        # self.locally = locally
        # self.create = create
        # self.path_volumes = path_volumes
        # self.path_colors = path_colors

        self.edge_elements = edge_elements
        self.plane_elements = plane_elements
        self.volume_elements = volume_elements
        self.void_dim = void_dim
        self.grid = grid
        self.step = int(self.void_dim / 6)
        self.directory = (str(directory),)
        self.n_samples = n_samples

        if step is not None:
            self.step = step

    def create_sculpts(
        self,
        # n_edge_elements=N_EDGE_ELEMENTS,
        # n_plane_elements=N_PLANE_ELEMENTS,
        # n_volume_elements=N_VOLUME_ELEMENTS,
        # verbose=os.environ.get("VERBOSE"),
        # void_dim=int(os.environ.get("VOID_DIM")),
    ):

        raw_data = []

        color_raw_data = []

        count = 0

        for count, sculpture in enumerate(range(self.n_samples)):  #

            if int(os.environ.get("VERBOSE")) == 1:
                print(
                    "\n⏹ "
                    + Fore.BLUE
                    + f"Creating sculpture number {count}"
                    + Style.RESET_ALL
                )

            start = time.time()

            if int(os.environ.get("VERBOSE")) == 1:
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
                void_dim=self.void_dim,
                edges=(
                    self.edge_elements[0],
                    self.edge_elements[1],
                    self.edge_elements[2],
                ),  # number of elements, minimun, maximun
                planes=(
                    self.plane_elements[0],
                    self.plane_elements[1],
                    self.plane_elements[2],
                ),
                volumes=(
                    self.volume_elements[0],
                    self.volume_elements[1],
                    self.volume_elements[2],
                ),
                grid=(
                    self.grid,
                    self.step,
                ),  # minimun height of column, and maximun height
                materials_edges=COLOR_EDGES,
                materials_planes=COLOR_PLANES,
                materials_volumes=COLOR_VOLUMES,
                step=self.step,
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
                    int(self.n_samples),
                    int(self.void_dim),
                    int(self.void_dim),
                    int(self.void_dim),
                )
            )
            .astype("int8")
        )

        color_raw_data = (
            np.asarray(color_raw_data)
            .reshape(
                (
                    int(self.n_samples),
                    int(self.void_dim),
                    int(self.void_dim),
                    int(self.void_dim),
                )
            )
            .astype("object")
        )

        Manager.make_directory(self.directory[0])

        np.save(
            f"{self.directory[0]}/volume_data[{date.today()}]",
            raw_data,
            allow_pickle=True,
        )

        np.save(
            f"{self.directory[0]}/material_data[{date.today()}]",
            color_raw_data,
            allow_pickle=True,
        )

        print(
            "\n🔽 "
            + Fore.BLUE
            + f"Just created 'volume_data' shaped {raw_data.shape} and 'material_data' shaped{color_raw_data.shape}"
            + Style.RESET_ALL
        )

        return (raw_data, color_raw_data)

    """def load_locally():
        raw_data = ''
        color_raw_data = ''
        print(
            "\n🔽 "
            + Fore.BLUE
            + f"Just Loaded 'raw_data' shaped {raw_data.shape} and 'color_raw_data' shaped{color_raw_data.shape} from computer"
            + Style.RESET_ALL
        )

    def load_from_gcp():
        raw_data = ''
        color_raw_data = ''
        print(
            "\n🔽 "
            + Fore.BLUE
            + f"Just Loaded 'raw_data' shaped {raw_data.shape} and 'color_raw_data' shaped{color_raw_data.shape} from gcp"
            + Style.RESET_ALL
        )

    def load_from_query():
        raw_data = ''
        color_raw_data = ''
        print(
            "\n🔽 "
            + Fore.BLUE
            + f"Just Loaded 'raw_data' shaped {raw_data.shape} and 'color_raw_data' shaped{color_raw_data.shape} from Big Query"
            + Style.RESET_ALL
        )"""


if __name__ == "__main__":

    out_dir = os.path.join(
        os.environ.get("HOME"), "code", "juan-garassino", "deepSculpt", "data"
    )

    curator = Curator(
        void_dim=32,  # locally=True, path_volumes="", path_colors="",
        edge_elements=(0, 0.3, 0.5),
        plane_elements=(0, 0.3, 0.5),
        volume_elements=(0, 0.3, 0.5),
        step=None,
        directory=out_dir,
        n_samples=100,
    )

    curator.create_sculpts()
