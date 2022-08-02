from deepSculpt.sculptor import Sculptor
from google.cloud import storage

import pandas as pd
import numpy as np
import os
from datetime import date

from deepSculpt.params import BUCKET_NAME, BUCKET_TRAIN_DATA_PATH, MODEL_BASE_PATH

class DataLoaderCreator():

    def __init__(self, locally=True, path_volumes='' ,path_colors=''):
        self.locally = locally
        self.path_volumes = path_volumes
        self.path_colors = path_colors

    def load(self, volumes="", colors=""):

        path = os.path.join(os.path.dirname(__file__), 'data')

        os.chdir(path)

        raw_data = np.load(volumes, allow_pickle=True)

        color_raw_data = np.load(colors, allow_pickle=True)

        print(
            f"Just loaded 'raw_data' shaped {raw_data.shape} and 'color_raw_data' shaped{color_raw_data.shape}"
        )

        return (raw_data, color_raw_data)

    def load_from_gcp(self):

        data_file = "data_train_1k.csv"

        client = storage.Client().bucket(BUCKET_NAME)

        blob = client.blob(BUCKET_TRAIN_DATA_PATH)

        blob.download_to_filename(data_file)

        df = pd.read_csv(data_file)

        return df

    def create(self,
               n_samples=5,
               n_edge_elements=0,
               n_plane_elements=0,
               n_volume_elements=0,
               color_edges="dimgrey",
               color_planes="snow",
               color_volumes = ["crimson", "turquoise", "gold"],
               verbose = False,
               void_dim=48):

        path = os.path.join(os.path.dirname(__file__), 'data')

        os.chdir(path)

        raw_data = []
        color_raw_data = []
        count = 0

        for sculpture in range(n_samples): #
            count = count + 1
            if count % 10 == 0:
                print("\r{0}".format(count), end='')

            sculptor = Sculptor(
                void_dim=void_dim,
                n_edge_elements=n_edge_elements,
                n_plane_elements=n_plane_elements,
                n_volume_elements=n_volume_elements,
                color_edges=color_edges,
                color_planes=color_planes,
                color_volumes=
                color_volumes,  # ["greenyellow","orange","mediumpurple"]
                element_edge_min=int(void_dim * 0.8),
                element_edge_max=int(void_dim * 0.9),
                element_grid_min=int(void_dim * 0.9),
                element_grid_max=int(void_dim * 0.95),
                element_plane_min=int(void_dim * 0.4),
                element_plane_max=int(void_dim * 0.8),
                element_volume_min=int(void_dim * 0.2),
                element_volume_max=int(void_dim * 0.5),
                step=1,
                verbose=verbose)

            sculpture = sculptor.generative_sculpt()

            raw_data.append(sculpture[0].astype("int8"))
            color_raw_data.append(sculpture[1])

        raw_data = np.asarray(raw_data).reshape(
            (n_samples, void_dim, void_dim, void_dim)).astype('int8')

        color_raw_data = np.asarray(color_raw_data).reshape(
            (n_samples, void_dim, void_dim, void_dim))

        np.save(f"raw-data[{date.today()}]",
                raw_data,
                allow_pickle=True)

        np.save(f"color-raw-data[{date.today()}]",
                color_raw_data,
                allow_pickle=True)

        print(
            f"Just created 'raw_data' shaped {raw_data.shape} and 'color_raw_data' shaped{color_raw_data.shape}"
        )

        return (raw_data, color_raw_data)

    def get_data(self):

        if self.locally:
            return self.load(volumes=self.path_volumes, colors=self.path_colors)
        else:
            return self.load_from_gcp()

    def clean_data(df):
        pass


    def holdout(df):
        pass

if __name__ == "__main__":
    df = get_data()
    print(df)
