# from xml.dom import NO_MODIFICATION_ALLOWED_ERR
import matplotlib.pyplot as plt
import numpy as np
from deepSculpt.manager.sculptor import Sculptor
from deepSculpt.manager.manager import Manager
from datetime import datetime
from colorama import Fore, Style


class Plotter(Sculptor):
    def __init__(
        self, volumes, colors, figsize=25, style="#ffffff", dpi=100, transparent=False
    ):

        self.void = volumes
        self.volumes = volumes
        self.colors = colors
        self.figsize = figsize
        self.style = style
        self.dpi = dpi
        self.transparent = transparent

    def plot_sections(self):
        sculpture = self.void
        fig, axes = plt.subplots(
            ncols=6,
            nrows=int(np.ceil(self.void.shape[0] / 6)),
            figsize=(self.figsize, self.figsize),
            facecolor=(self.style),
            dpi=self.dpi,
        )

        axes = axes.ravel()  # flats
        for index in range(self.void.shape[0]):
            axes[index].imshow(sculpture[index, :, :], cmap="gray")

    def plot_sculpture(
        self, directory
    ):  # add call to generative sculpt and then plot like 12
        fig, axes = plt.subplots(
            ncols=2,
            nrows=2,
            figsize=(self.figsize, self.figsize),
            facecolor=(self.style),
            subplot_kw=dict(projection="3d"),
            dpi=self.dpi,
        )

        axes = axes.ravel()  # flats

        # print(axes)
        # print(len(axes))

        for plot in range(1):  # to print one color i need a condition to not rotate!!
            axes[0].voxels(
                self.volumes,
                edgecolors="k",
                linewidth=0.05,
                facecolors=self.colors,
            )

            axes[1].voxels(
                np.rot90(self.volumes, 1),
                facecolors=np.rot90(self.colors, 1),
                edgecolors="k",
                linewidth=0.05,
            )

            axes[2].voxels(
                np.rot90(self.volumes, 2),
                facecolors=np.rot90(self.colors, 2),
                edgecolors="k",
                linewidth=0.05,
            )

            axes[3].voxels(
                np.rot90(self.volumes, 3),
                facecolors=np.rot90(self.colors, 3),
                edgecolors="k",
                linewidth=0.05,
            )

        Manager.make_directory(directory)

        now = datetime.now().strftime("%d-%m-%Y-%H-%M")

        name_png = f"{directory}/image[{now}].png"

        plt.savefig(
            name_png, transparent=self.transparent
        )  # agregar tiempo de impresion y exportar 3D y bounding box

        print(
            "\n🔽 "
            + Fore.BLUE
            + f"Just created a snapshot {name_png.split('/')[-1]} @ {directory}"
            + Style.RESET_ALL
        )

        name_svg = f"{directory}/vectorial[{now}].svg"

        plt.savefig(name_svg, transparent=self.transparent)

        print(
            "\n🔽 "
            + Fore.BLUE
            + f"Just created a vectorial snapshot {name_svg.split('/')[-1]} @ {directory}"
            + Style.RESET_ALL
        )
