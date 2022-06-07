import matplotlib.pyplot as plt

class Plotter():

    def __init__(self, void):
        self.void = void

    def plot_sections(self):
        sculpture = self.void
        fig, axes = plt.subplots(ncols=6,
                                 nrows=int(np.ceil(self.void.shape[0] / 6)),
                                 figsize=(25, 25),
                                 facecolor=(self.style))
        axes = axes.ravel()  # flats
        for index in range(self.void.shape[0]):
            axes[index].imshow(sculpture[index, :, :], cmap="gray")

    def plot_sculpture(
            self):  # add call to generative sculpt and then plot like 12
        #sculpture = self.void
        fig, axes = plt.subplots(ncols=2,
                                 nrows=2,
                                 figsize=(25, 25),
                                 facecolor=(self.style),
                                 subplot_kw=dict(projection="3d"))
        axes = axes.ravel()  # flats
        for index in range(1):
            self.void = np.zeros((void_dim, void_dim, void_dim))
            sculpture = self.generative_sculpt()
            axes[index].voxels(sculpture[0],
                               facecolors=sculpture[1],
                               edgecolors="k",
                               linewidth=0.05)  # axes[index]
        plt.savefig('image.png')  # agregar tiempo de impresion y exportar 3D
