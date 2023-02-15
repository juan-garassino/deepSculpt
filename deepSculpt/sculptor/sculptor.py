from deepSculpt.sculptor.components.cantilever import attach_pipe
from deepSculpt.sculptor.components.edge_components import attach_edge
from deepSculpt.sculptor.components.grid_components import attach_grid
from deepSculpt.sculptor.components.plane_components import attach_plane
from deepSculpt.curator.tools.params import COLORS

import numpy as np
from colorama import Fore, Style
import time
import os


class Sculptor:
    def __init__(
        self,
        void_dim=16,
        edges=(1, 0.3, 0.5),
        planes=(1, 0.3, 0.5),
        volumes=(1, 0.3, 0.5),
        grid=(1, 4),
        materials_edges=None,
        materials_planes=None,
        materials_volumes=None,
        step=1,
    ):
        """
        Creates one sculpt

        parameters: edges (numbers of elements, minimun size, maximun size)
        """
        self.void_dim = void_dim
        self.volumes_void = np.zeros(
            (self.void_dim, self.void_dim, self.void_dim)
        )  # Creates a void
        self.materials_void = np.empty(
            self.volumes_void.shape, dtype=object
        )  # Creates a color void

        self.color_edges = materials_edges  # list of colors for the edges
        self.color_planes = materials_planes  # list of colors for the planes
        self.color_volumes = materials_volumes  # list of colors for the volumes

        self.n_edge_elements = edges[0]
        self.n_plane_elements = planes[0]
        self.n_volume_elements = volumes[0]
        self.style = "#ffffff"

        self.element_edge_min = edges[1]
        self.element_edge_max = edges[2]

        self.element_plane_min = planes[1]
        self.element_plane_max = planes[2]

        self.element_volume_min = volumes[1]
        self.element_volume_max = volumes[2]

        self.step = step
        self.grid = grid

    def generative_sculpt(self):
        start = time.time()

        if self.grid[0] == 1:
            for grid in range(1):

                if int(os.environ.get("VERBOSE")) == 1:
                    print("\n ⏹  " + Fore.MAGENTA + "Creating grid" + Style.RESET_ALL)

                attach_grid(
                    volumes_void=self.volumes_void,
                    materials_void=self.materials_void,
                    step=self.grid[1],
                    verbose=int(os.environ.get("VERBOSE")),
                )

        for edge in range(self.n_edge_elements):

            if int(os.environ.get("VERBOSE")) == 1:
                print(
                    "\n ⏹  "
                    + Fore.MAGENTA
                    + f"Creating edge number {edge}"
                    + Style.RESET_ALL
                )

            attach_edge(
                self.volumes_void,
                self.materials_void,
                element_edge_min_ratio=self.element_edge_min,
                element_edge_max_ratio=self.element_edge_max,
                step=self.step,
                verbose=int(os.environ.get("VERBOSE")),
            )

        for plane in range(self.n_plane_elements):

            if int(os.environ.get("VERBOSE")) == 1:
                print(
                    "\n ⏹  "
                    + Fore.MAGENTA
                    + f"Creating plane number {plane}"
                    + Style.RESET_ALL
                )

            attach_plane(
                self.volumes_void,
                self.materials_void,
                element_plane_min_ratio=self.element_plane_min,
                element_plane_max_ratio=self.element_plane_max,
                step=self.step,
                verbose=int(os.environ.get("VERBOSE")),
            )

        for volume in range(self.n_volume_elements):

            if int(os.environ.get("VERBOSE")) == 1:
                print(
                    "\n ⏹  "
                    + Fore.MAGENTA
                    + f"Creating volume number {volume}"
                    + Style.RESET_ALL
                )

            attach_pipe(
                self.volumes_void,
                self.materials_void,
                self.element_volume_min,
                self.element_volume_max,
                self.step,
            )

        if int(os.environ.get("VERBOSE")) == 1:
            print(
                "\n ⏹  "
                + Fore.GREEN
                + "Time for sculptures is {} sec".format(time.time() - start)
                + Style.RESET_ALL
            )

        return self.volumes_void, self.materials_void


if __name__ == "__main__":

    sculptor = Sculptor(
        void_dim=16,
        edges=(1, 0.3, 0.5),  # number of elements, minimun, maximun
        planes=(1, 0.3, 0.55),
        volumes=(1, 0.7, 0.8),
        grid=(2, 5),  # minimun height of column, and maximun height
        materials_edges=COLORS["edges"],
        materials_planes=COLORS["planes"],
        materials_volumes=COLORS["volumes"],
        step=1,
    )

    sculpt = sculptor.generative_sculpt()

    print(sculpt)
