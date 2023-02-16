from deepSculpt.manager.tools.params import COLORS

from deepSculpt.sculptor.components.cantilever import attach_pipe
from deepSculpt.sculptor.components.edge_components import attach_edge
from deepSculpt.sculptor.components.grid_components import attach_grid
from deepSculpt.sculptor.components.plane_components import attach_plane

from typing import List, Tuple
import numpy as np
from colorama import Fore, Style
import time
import os


class Sculptor:
    def __init__(
        self,
        void_dim: int = 16,
        edges: Tuple[int, float, float] = (1, 0.3, 0.5),
        planes: Tuple[int, float, float] = (1, 0.3, 0.5),
        volumes: Tuple[int, float, float] = (1, 0.3, 0.5),
        grid: Tuple[int, int] = (1, 4),
        materials_edges: List[str] = None,
        materials_planes: List[str] = None,
        materials_volumes: List[str] = None,
        step: int = 1,
    ):  # -> None:
        """
        Initializes a new Sculptor instance.

        Args:
            void_dim:
            The dimension of the void to be created (default: 16).
            edges:
            A tuple of three values specifying the number of elements, minimum size, and maximum size of the edges (default: (1, 0.3, 0.5)).
            planes:
            A tuple of three values specifying the number of elements, minimum size, and maximum size of the planes (default: (1, 0.3, 0.5)).
            volumes:
            A tuple of three values specifying the number of elements, minimum size, and maximum size of the volumes (default: (1, 0.3, 0.5)).
            grid:
            A tuple of two values specifying the minimum height of a column and the maximum height of a column in the grid (default: (1, 4)).
            materials_edges:
            A list of colors for the edges (default: None).
            materials_planes:
            A list of colors for the planes (default: None).
            materials_volumes:
            A list of colors for the volumes (default: None).
            step:
            The step size used in creating the components (default: 1).
        """
        self.void_dim = void_dim
        # Creates a void
        self.volumes_void = np.zeros((self.void_dim, self.void_dim, self.void_dim))
        # Creates a color void
        self.materials_void = np.empty(self.volumes_void.shape, dtype=object)

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

    def generative_sculpt(self):  # -> tuple[np.ndarray, np.ndarray]:
        """
        Generates a sculpture by attaching edge, plane and volume components

        Returns:
        tuple[np.ndarray, np.ndarray]: The sculpted volumes and materials
        """
        # Initialize a timer for the entire process
        start = time.time()

        # Create grid components
        if self.grid[0] == 1:
            for grid in range(1):
                # Print status if verbose is on
                if int(os.environ.get("VERBOSE")) == 1:
                    print("\n ⏹  " + Fore.MAGENTA + "Creating grid" + Style.RESET_ALL)

                # Attach grid components
                attach_grid(
                    volumes_void=self.volumes_void,
                    materials_void=self.materials_void,
                    step=self.grid[1],
                    verbose=int(os.environ.get("VERBOSE")),
                )

        # Create edge components
        for edge in range(self.n_edge_elements):
            # Print status if verbose is on
            if int(os.environ.get("VERBOSE")) == 1:
                print(
                    "\n ⏹  "
                    + Fore.MAGENTA
                    + f"Creating edge number {edge}"
                    + Style.RESET_ALL
                )

            # Attach edge components
            attach_edge(
                self.volumes_void,
                self.materials_void,
                element_edge_min_ratio=self.element_edge_min,
                element_edge_max_ratio=self.element_edge_max,
                step=self.step,
                verbose=int(os.environ.get("VERBOSE")),
            )

        # Create plane components
        for plane in range(self.n_plane_elements):
            # Print status if verbose is on
            if int(os.environ.get("VERBOSE")) == 1:
                print(
                    "\n ⏹  "
                    + Fore.MAGENTA
                    + f"Creating plane number {plane}"
                    + Style.RESET_ALL
                )

            # Attach plane components
            attach_plane(
                self.volumes_void,
                self.materials_void,
                element_plane_min_ratio=self.element_plane_min,
                element_plane_max_ratio=self.element_plane_max,
                step=self.step,
                verbose=int(os.environ.get("VERBOSE")),
            )

        # Create volume components
        for volume in range(self.n_volume_elements):
            # Print status if verbose is on
            if int(os.environ.get("VERBOSE")) == 1:
                print(
                    "\n ⏹  "
                    + Fore.MAGENTA
                    + f"Creating volume number {volume}"
                    + Style.RESET_ALL
                )

            # Attach volume components
            attach_pipe(
                self.volumes_void,
                self.materials_void,
                self.element_volume_min,
                self.element_volume_max,
                self.step,
            )

        # Print elapsed time if verbose is on
        if int(os.environ.get("VERBOSE")) == 1:
            print(
                "\n ⏹  "
                + Fore.GREEN
                + "Time for sculptures is {} sec".format(time.time() - start)
                + Style.RESET_ALL
            )

        # Return the numpy arrays of volumes and materials
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
