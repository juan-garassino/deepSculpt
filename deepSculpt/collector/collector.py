import os
import random
from datetime import date
from typing import List, Tuple
import time
import numpy as np
from colorama import Fore, Style

from deepSculpt.sculptor.sculptor import Sculptor
from deepSculpt.manager.manager import Manager
from deepSculpt.manager.tools.plotter import Plotter
from deepSculpt.collector.tools.params import COLORS


class Collector:
    def __init__(
        self,
        void_dim: int = 32,
        edge_elements: Tuple[float, float, float] = (0, 0.3, 0.5),
        plane_elements: Tuple[float, float, float] = (0, 0.3, 0.5),
        volume_elements: Tuple[float, float, float] = (0, 0.3, 0.5),
        step: int = None,
        directory: str = None,
        n_samples: int = 100,
        grid: int = 1,
    ):  # -> None:
        """Initialize the Curator instance.

        Args:
            void_dim (int, optional):
            The size of the 3D grid in each dimension. Defaults to 32.
            edge_elements (Tuple[float, float, float], optional):
            The tuple containing the number of edges and the minimum and maximum number of edges for a shape. Defaults to (0, 0.3, 0.5).
            plane_elements (Tuple[float, float, float], optional):
            The tuple containing the number of planes and the minimum and maximum number of planes for a shape. Defaults to (0, 0.3, 0.5).
            volume_elements (Tuple[float, float, float], optional):
            The tuple containing the number of volumes and the minimum and maximum number of volumes for a shape. Defaults to (0, 0.3, 0.5).
            step (int, optional):
            The step size for the 3D grid. Defaults to None.
            directory (str, optional):
            The directory path where the data files will be saved. Defaults to None.
            n_samples (int, optional):
            The number of samples to generate. Defaults to 100.
            grid (int, optional):
            The minimum height of a column and the maximum height of a column on the 3D grid. Defaults to 1.
        """
        self.void_dim = void_dim
        self.edge_elements = edge_elements
        self.plane_elements = plane_elements
        self.volume_elements = volume_elements
        self.grid = grid
        self.step = int(self.void_dim / 6) if step is None else step
        self.directory = str(directory) if directory is not None else None
        self.n_samples = n_samples

    def create_collection(self):  # -> Tuple[np.ndarray, np.ndarray]:
        """Generate the 3D sculpted shapes.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two NumPy arrays, the first one is a
            4D NumPy array of the volume data, and the second one is a 4D NumPy array of the material
            data of the generated shapes.
        """
        volumes_raw_data: List[np.ndarray] = []

        materials_raw_data: List[np.ndarray] = []

        count = 0

        for count, sculpture in enumerate(range(self.n_samples)):  #

            if int(os.environ.get("VERBOSE")) == 1:
                print(
                    "\n\t⏹ "
                    + Fore.BLUE
                    + f"Creating sculpture number {count}"
                    + Style.RESET_ALL
                )

            start = time.time()

            if int(os.environ.get("VERBOSE")) == 1:
                if (count + 1) % 25 == 0:
                    print(
                        "\n\t⏹ "
                        + Fore.GREEN
                        + "{} sculputers where created in {}".format(
                            (count + 1), time.time() - start
                        )
                        + Style.RESET_ALL
                    )

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
                materials_edges=COLORS["edges"],
                materials_planes=COLORS["planes"],
                materials_volumes=COLORS["volumes"],
                step=self.step,
            )

            sculpture = sculptor.generative_sculpt()

            volumes_raw_data.append(
                sculpture[0].astype("int8")
            )  # NOT APPEND BUT SAVE IN DIFF FILES!!

            materials_raw_data.append(sculpture[1])

        volumes_raw_data = (
            np.asarray(volumes_raw_data)
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

        materials_raw_data = (
            np.asarray(materials_raw_data)
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

        Manager.make_directory(self.directory)

        np.save(
            f"{self.directory}/volume_data[{date.today()}]",
            volumes_raw_data,
            allow_pickle=True,
        )

        np.save(
            f"{self.directory}/material_data[{date.today()}]",
            materials_raw_data,
            allow_pickle=True,
        )

        print(
            "\n 🔽 "
            + Fore.BLUE
            + f"Just created 'volume_data' shaped {volumes_raw_data.shape} and 'material_data' shaped{materials_raw_data.shape}"
            + Style.RESET_ALL
        )

        # path
        if int(os.environ.get("INSTANCE")) == 0:
            path = os.path.join(
                os.environ.get("HOME"),
                "code",
                "juan-garassino",
                "deepSculpt",
                "data",
                "sampling",
            )
        # path
        if int(os.environ.get("INSTANCE")) == 1:
            path = os.path.join(
                os.environ.get("HOME"),
                "..",
                "content",
                "drive",
                "MyDrive",
                "repositories",
                "deepSculpt",
                "data",
                "sampling",
            )

        for sample in range(int(os.environ.get("N_SAMPLES_PLOT"))):

            index = random.choices(list(np.arange(0, self.n_samples, 1)), k=1)[0]

            Plotter(
                volumes_raw_data[index],
                materials_raw_data[index],
                figsize=25,
                style="#ffffff",
                dpi=int(os.environ.get("DPI")),
            ).plot_sculpture(path)

            print(
                "\n 🆗 "
                + Fore.YELLOW
                + f"Just ploted 'volume_data[{index}]' and 'material_data[{index}]'"
                + Style.RESET_ALL
            )

        return (volumes_raw_data, materials_raw_data)


if __name__ == "__main__":

    out_dir = os.path.join(
        os.environ.get("HOME"), "code", "juan-garassino", "deepSculpt", "data"
    )

    collector = Collector(
        void_dim=int(os.environ.get("VOID_DIM")),
        edge_elements=(0, 0.3, 0.5),
        plane_elements=(0, 0.3, 0.5),
        volume_elements=(2, 0.3, 0.5),
        step=None,
        directory=out_dir,
        n_samples=50,
        grid=1,
    )

    collector.create_collection()
