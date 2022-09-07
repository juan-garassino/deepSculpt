from deepSculpt.manager.components.cantilever import add_pipe_cantilever
from deepSculpt.manager.components.edges import add_edge
from deepSculpt.manager.components.grid import add_grid
from deepSculpt.manager.components.planes import add_plane

import time
import numpy as np


class Sculptor:
    def __init__(
        self,
        void_dim,
        n_edge_elements,
        n_plane_elements,
        n_volume_elements,
        color_edges,
        color_planes,
        color_volumes,
        element_edge_min,
        element_edge_max,
        element_grid_min,
        element_grid_max,
        element_plane_min,
        element_plane_max,
        element_volume_min,
        element_volume_max,
        step,
        verbose,
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

        print("Time for sculptures is {} sec".format(time.time() - start))

        return self.void, self.color_void
