import matplotlib.pyplot as plt
import numpy as np
from deepSculpt.sculptor import Sculptor

class Plotter(Sculptor):

    def __init__(self, void, style):
        self.void = void
        self.style = style

    def plot_sections(self):
        sculpture = self.void
        fig, axes = plt.subplots(ncols=6,
                                 nrows=int(np.ceil(self.void.shape[0] / 6)),
                                 figsize=(25, 25),
                                 facecolor=(self.style))
        axes = axes.ravel()  # flats
        for index in range(self.void.shape[0]):
            axes[index].imshow(sculpture[index, :, :], cmap="gray")

    def plot_sculpture(self):  # add call to generative sculpt and then plot like 12
        fig, axes = plt.subplots(ncols=2,
                                 nrows=1,
                                 figsize=(25, 25),
                                 facecolor=(self.style),
                                 subplot_kw=dict(projection="3d"))
        axes = axes.ravel()  # flats
        for index in range(1):
            axes[0].voxels(self.void[0],
                               facecolors=self.void[1],
                               edgecolors="k",
                               linewidth=0.05)
            axes[1].voxels(self.void[0].T,
                               facecolors=self.void[1].T,
                               edgecolors="k",
                               linewidth=0.05)  # axes[index]
        plt.savefig('image.png')  # agregar tiempo de impresion y exportar 3D
