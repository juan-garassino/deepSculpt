from deepSculpt.sculptor import Sculptor
from deepSculpt.params import (
    VOID_DIM,
    N_SAMPLES,
    BUCKET_NAME,
    BUCKET_TRAIN_DATA_PATH,
    MODEL_BASE_PATH,
    N_EDGE_ELEMENTS,
    N_PLANE_ELEMENTS,
    N_VOLUME_ELEMENTS,
    COLOR_EDGES,
    COLOR_PLANES,
    COLOR_VOLUMES,
    ELEMENT_EDGE_MIN,
    ELEMENT_EDGE_MAX,
    ELEMENT_GRID_MIN,
    ELEMENT_GRID_MAX,
    ELEMENT_PLANE_MIN,
    ELEMENT_PLANE_MAX,
    ELEMENT_VOLUME_MIN,
    ELEMENT_VOLUME_MAX,
    VERBOSE,
)

from datetime import date
import numpy as np
import os
from google.cloud import storage

class DataLoaderCreator:
    def __init__(self, create=False, locally=True, path_volumes="", path_colors=""):
        self.locally = locally
        self.create = create
        self.path_volumes = path_volumes
        self.path_colors = path_colors

    def create_sculpts(
        self,
        n_samples=N_SAMPLES,
        n_edge_elements=N_EDGE_ELEMENTS,
        n_plane_elements=N_PLANE_ELEMENTS,
        n_volume_elements=N_VOLUME_ELEMENTS,
        color_edges=COLOR_EDGES,
        color_planes=COLOR_PLANES,
        color_volumes=COLOR_VOLUMES,
        verbose=VERBOSE,
        void_dim=VOID_DIM,
    ):

        path = os.path.join(os.path.dirname(__file__), "data")

        os.chdir(path)

        raw_data = []
        color_raw_data = []
        count = 0

        for sculpture in range(n_samples):  #
            count = count + 1
            if count % 10 == 0:
                print("\r{0}".format(count), end="")

            sculptor = Sculptor(
                void_dim=void_dim,
                n_edge_elements=n_edge_elements,
                n_plane_elements=n_plane_elements,
                n_volume_elements=n_volume_elements,
                color_edges=color_edges,
                color_planes=color_planes,
                color_volumes=color_volumes,  # ["greenyellow","orange","mediumpurple"]
                element_edge_min=ELEMENT_EDGE_MIN,
                element_edge_max=ELEMENT_EDGE_MAX,
                element_grid_min=ELEMENT_GRID_MIN,
                element_grid_max=ELEMENT_GRID_MAX,
                element_plane_min=ELEMENT_PLANE_MIN,
                element_plane_max=ELEMENT_PLANE_MAX,
                element_volume_min=ELEMENT_VOLUME_MIN,
                element_volume_max=ELEMENT_VOLUME_MAX,
                step=1,
                verbose=verbose,
            )

            sculpture = sculptor.generative_sculpt()

            raw_data.append(sculpture[0].astype("int8"))
            color_raw_data.append(sculpture[1])

        raw_data = (
            np.asarray(raw_data)
            .reshape((N_SAMPLES, VOID_DIM, VOID_DIM, VOID_DIM))
            .astype("int8")
        )

        color_raw_data = np.asarray(color_raw_data).reshape(
            (N_SAMPLES, VOID_DIM, VOID_DIM, VOID_DIM)
        )

        np.save(f"raw-data[{date.today()}]", raw_data, allow_pickle=True)

        np.save(f"color-raw-data[{date.today()}]", color_raw_data, allow_pickle=True)

        print(
            f"Just created 'raw_data' shaped {raw_data.shape} and 'color_raw_data' shaped{color_raw_data.shape}"
        )

        return (raw_data, color_raw_data)

    def load_locally(self):

        path = os.path.join(os.path.dirname(__file__), "data")

        os.chdir(path)

        raw_data = np.load(self.path_volumes, allow_pickle=True)

        color_raw_data = np.load(self.path_colors, allow_pickle=True)

        print(
            f"Just loaded 'raw_data' shaped {raw_data.shape} and 'color_raw_data' shaped{color_raw_data.shape}"
        )

        return (raw_data, color_raw_data)

    def load_from_gcp(self):

        files = [self.path_volumes, self.path_colors]

        client = storage.Client().bucket(BUCKET_NAME)

        for file in files:

            blob = client.blob(BUCKET_TRAIN_DATA_PATH + '/' + file)

            blob.download_to_filename(file)

        raw_data = np.load(self.path_volumes, allow_pickle=True)

        color_raw_data = np.load(self.path_colors, allow_pickle=True)

        print(
            f"Just loaded 'raw_data' shaped {raw_data.shape} and 'color_raw_data' shaped{color_raw_data.shape}"
        )

        return (raw_data, color_raw_data)#, color_raw_data)

    def clean_data(df):
        pass

    def holdout(df):
        pass


if __name__ == "__main__":

    data = DataLoaderCreator(create=False,
                             locally=True,
                             path_volumes="raw-data[2022-07-26].npy",
                             path_colors="color-raw-data[2022-07-26].npy").load_from_gcp()

    data[0].shape
