from deepSculpt.sculptor.components.cantilever import add_pipe_cantilever
from deepSculpt.sculptor.components.edges import add_edge
from deepSculpt.sculptor.components.grid import add_grid
from deepSculpt.sculptor.components.planes import add_plane

import time
import numpy as np
from colorama import Fore, Style


class Sculptor:
    def __init__(
        self,
        void_dim=16,
        n_edge_elements=1,
        n_plane_elements=1,
        n_volume_elements=1,
        color_edges=None,
        color_planes=None,
        color_volumes=None,
        element_edge_min=1,
        element_edge_max=5,
        element_grid_min=1,
        element_grid_max=5,
        element_plane_min=1,
        element_plane_max=5,
        element_volume_min=1,
        element_volume_max=5,
        step=1,
        verbose=False,
    ):

        self.void = np.zeros((void_dim, void_dim, void_dim))
        self.color_void = np.empty(self.void.shape, dtype=object)
        self.colors = np.empty(self.void.shape, dtype=object)

        self.color_edges = color_edges
        self.color_planes = color_planes
        self.color_volumes = color_volumes

        self.n_edge_elements = n_edge_elements
        self.n_plane_elements = n_plane_elements
        self.n_volume_elements = n_volume_elements
        self.style = "#ffffff"

        self.element_edge_min = element_edge_min
        self.element_edge_max = element_edge_max
        self.element_grid_min = element_grid_min
        self.element_grid_max = element_grid_max
        self.element_plane_min = element_plane_min
        self.element_plane_max = element_plane_max
        self.element_volume_min = element_volume_min
        self.element_volume_max = element_volume_max
        self.step = step

        self.verbose = verbose

    def generative_sculpt(self):
        start = time.time()
        for edge in range(self.n_edge_elements):
            add_edge(
                self.void,
                self.color_void,
                self.element_edge_min,
                self.element_edge_max,
                self.step,
                self.verbose,
            )

        for plane in range(self.n_plane_elements):
            add_plane(
                self.void,
                self.color_void,
                self.element_plane_min,
                self.element_plane_max,
                self.step,
                self.verbose,
            )

        for volume in range(self.n_volume_elements):
            add_pipe_cantilever(
                self.void,
                self.color_void,
                self.element_volume_min,
                self.element_volume_max,
                self.step,
                self.verbose,
            )

        for volume in range(self.n_volume_elements):
            add_grid(
                self.void,
                self.color_void,
                self.element_grid_min,
                self.element_grid_max,
                self.step,
                self.verbose,
            )

        if self.verbose:
            print(
                "\n⏹  "
                + Fore.GREEN
                + "Time for sculptures is {} sec".format(time.time() - start)
                + Style.RESET_ALL
            )

        return self.void, self.color_void

if __name__ == "__main__":

    sculptor = Sculptor(
                        void_dim=16,
                        n_edge_elements=1,
                        n_plane_elements=1,
                        n_volume_elements=1,
                        color_edges=None,
                        color_planes=None,
                        color_volumes=None,
                        element_edge_min=2,
                        element_edge_max=5,
                        element_grid_min=2,
                        element_grid_max=5,
                        element_plane_min=2,
                        element_plane_max=5,
                        element_volume_min=2,
                        element_volume_max=5,
                        step=1,
                        verbose=False,
    )

    sculpt = sculptor.generative_sculpt()

    print(sculpt)
