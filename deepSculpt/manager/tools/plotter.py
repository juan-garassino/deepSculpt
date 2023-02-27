# from xml.dom import NO_MODIFICATION_ALLOWED_ERR
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from deepSculpt.sculptor.sculptor import Sculptor
from deepSculpt.manager.manager import Manager
from datetime import datetime
from colorama import Fore, Style


class Plotter(Sculptor):
    def __init__(
        self,
        volumes=None,
        colors=None,
        figsize=25,
        style="#ffffff",
        dpi=100,
        transparent=False,
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
        self,
        directory,
        raster_picture=False,
        vector_picture=False,
        volumes_array=False,
        materials_array=False,
        hide_axis=False,
    ):  # add call to generative sculpt and then plot like 12
        fig, axes = plt.subplots(
            ncols=2,
            nrows=2,
            figsize=(self.figsize, self.figsize),
            facecolor=(self.style),
            subplot_kw=dict(projection="3d"),
            dpi=self.dpi,
        )

        axes = axes.ravel()

        if type(self.colors).__module__ == np.__name__:
            for _ in range(1):

                if hide_axis:
                    axes[0].set_axis_off()

                axes[0].voxels(
                    self.volumes,
                    edgecolors="k",
                    linewidth=0.05,
                    facecolors=self.colors,
                )

                if hide_axis:
                    axes[1].set_axis_off()

                axes[1].voxels(
                    np.rot90(self.volumes, 1),
                    facecolors=np.rot90(self.colors, 1),
                    edgecolors="k",
                    linewidth=0.05,
                )

                if hide_axis:
                    axes[2].set_axis_off()

                axes[2].voxels(
                    np.rot90(self.volumes, 2),
                    facecolors=np.rot90(self.colors, 2),
                    edgecolors="k",
                    linewidth=0.05,
                )

                if hide_axis:
                    axes[3].set_axis_off()

                axes[3].voxels(
                    np.rot90(self.volumes, 3),
                    facecolors=np.rot90(self.colors, 3),
                    edgecolors="k",
                    linewidth=0.05,
                )

        else:
            for _ in range(1):
                axes[0].voxels(
                    self.volumes,
                    edgecolors="k",
                    linewidth=0.05,
                )

                axes[1].voxels(
                    np.rot90(self.volumes, 1),
                    edgecolors="k",
                    linewidth=0.05,
                )

                axes[2].voxels(
                    np.rot90(self.volumes, 2),
                    edgecolors="k",
                    linewidth=0.05,
                )

                axes[3].voxels(
                    np.rot90(self.volumes, 3),
                    edgecolors="k",
                    linewidth=0.05,
                )

        now = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

        print("\n 🔽 " + Fore.GREEN + f"Plotting [{now}]" + Style.RESET_ALL)

        if raster_picture:

            Manager.make_directory(directory + "/picture")

            name_png = f"{directory}/picture/image[{now}].png"

            plt.savefig(name_png, transparent=self.transparent)

            print(
                "\n ✅ "
                + Fore.BLUE
                + f"Just created a snapshot {name_png.split('/')[-1]} @ {directory  + '/picture'}"
                + Style.RESET_ALL
            )

        if vector_picture:

            Manager.make_directory(directory + "/vectorial")

            name_svg = f"{directory}/vectorial/vectorial[{now}].svg"

            plt.savefig(name_svg, transparent=self.transparent)

            print(
                "\n ✅ "
                + Fore.BLUE
                + f"Just created a vectorial snapshot {name_svg.split('/')[-1]} @ {directory  + '/vectorial'}"
                + Style.RESET_ALL
            )

        if volumes_array:

            Manager.make_directory(directory + "/volume_array")

            name_volume_array = f"{directory}/volume_array/volume_array[{now}]"

            np.save(name_volume_array, self.volumes)

            print(
                "\n ✅ "
                + Fore.BLUE
                + f"Just created a volume array {name_volume_array.split('/')[-1]} @ {directory + '/volume_array'}"
                + Style.RESET_ALL
            )

        if materials_array:

            Manager.make_directory(directory + "/material_array")

            name_material_array = f"{directory}/material_array/material_array[{now}]"

            np.save(name_material_array, self.colors)

            print(
                "\n ✅ "
                + Fore.BLUE
                + f"Just created a material array {name_material_array.split('/')[-1]} @ {directory + '/material_array'}"
                + Style.RESET_ALL
            )

    @staticmethod
    def voxel_to_pointscloud(arr, N):
        n_x, n_y, n_z = arr.shape
        new_arr = np.zeros((N * n_x, N * n_y, N * n_z, 3))
        for i in range(n_x):
            for j in range(n_y):
                for k in range(n_z):
                    if arr[i, j, k]:
                        x = np.linspace(i, i + 1, N + 1)[:-1]
                        y = np.linspace(j, j + 1, N + 1)[:-1]
                        z = np.linspace(k, k + 1, N + 1)[:-1]
                        xv, yv, zv = np.meshgrid(x, y, z, indexing="ij")
                        vertices = np.stack((xv, yv, zv), axis=-1)
                        vertices = vertices.reshape(-1, 3)
                        new_arr[
                            N * i : N * (i + 1),
                            N * j : N * (j + 1),
                            N * k : N * (k + 1),
                        ] = vertices.reshape(N, N, N, 3)
        return np.unique(new_arr.reshape(-1, 3), axis=0)

    @staticmethod
    def plot_pointscloud(x, y, z, size=1.0, color=(0, 0, 0), alpha=1.0):
        # Create the scatter3d trace
        trace = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            marker=dict(
                size=size, color=f"rgba({color[0]}, {color[1]}, {color[2]}, {alpha})"
            ),
        )

        # Set the layout of the plot
        layout = go.Layout(
            scene=dict(
                aspectratio=dict(x=1, y=1, z=1),
                xaxis=dict(title="X"),
                yaxis=dict(title="Y"),
                zaxis=dict(title="Z"),
            )
        )

        # Plot the trace
        fig = go.Figure(data=trace, layout=layout)

        fig.update_layout(width=1200, height=800)

        fig.show()
