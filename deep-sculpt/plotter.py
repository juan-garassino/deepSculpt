import matplotlib.pyplot as plt

class Plotter():

    def __init__(self, void):
        self.void = void

    def plot_sections(self):
        sculpture = self.void
        fig, axes = plt.subplots(ncols=6, nrows=int(np.ceil(self.void.shape[0]/6)), figsize=(25, 25), facecolor = (self.style))
        axes = axes.ravel() # flats
        for index in range(self.void.shape[0]):
            axes[index].imshow(sculpture[index,:,:], cmap = "gray")

    def plot_sculpture(self):
        sculpture = self.void
        fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(25, 25), facecolor = (self.style), subplot_kw=dict(projection="3d"))
        axes = axes.ravel() # flats
        for index in range(1):
            axes[index].voxels(sculpture, facecolors="orange", edgecolors="k", linewidth=0.05)
