from matplotlib import pyplot as plt
from colorama import Fore, Style
from google.cloud import storage
from tensorflow.data import Dataset
import os
import numpy as np
import errno


class Manager:  # make manager work with and with out epochs
    def __init__(self, model_name, data_name):
        self.model_name = model_name

        self.data_name = data_name

        self.comment = "{}_{}".format(model_name, data_name)

        self.data_subdir = "{}/{}".format(model_name, data_name)

    def upload_snapshot_to_gcp(snapshot_name):

        STORAGE_FILENAME = snapshot_name

        storage_location = f"results/{STORAGE_FILENAME}"

        bucket = storage.Client().bucket(os.environ.get("BUCKET_NAME"))

        blob = bucket.blob(storage_location)

        blob.upload_from_filename(STORAGE_FILENAME)

        print(
            "\n🔼 "
            + Fore.BLUE
            + f"Just uploaded a snapshot to gcp {STORAGE_FILENAME} @ {storage_location}"
            + Style.RESET_ALL
        )

    def load_locally(self):

        raw_data = np.load(self.path_volumes, allow_pickle=True)[
            : int(os.environ.get("TRAIN_SIZE"))
        ]

        color_raw_data = np.load(self.path_colors, allow_pickle=True)[
            : int(os.environ.get("TRAIN_SIZE"))
        ]

        print(
            "\n🔼 "
            + Fore.BLUE
            + f"Just loaded 'raw_data' shaped {raw_data.shape} and 'color_raw_data' shaped{color_raw_data.shape}"
            + Style.RESET_ALL
        )

        return (raw_data, color_raw_data)

    def load_from_gcp(self):

        files = [self.path_volumes, self.path_colors]

        client = storage.Client().bucket(os.environ.get("BUCKET_NAME"))

        for file in files:

            blob = client.blob(os.environ.get("BUCKET_TRAIN_DATA_PATH") + "/" + file)

            blob.download_to_filename(file)

        raw_data = np.load(self.path_volumes, allow_pickle=True)[
            : int(os.environ.get("TRAIN_SIZE"))
        ]

        color_raw_data = np.load(self.path_colors, allow_pickle=True)[
            : int(os.environ.get("TRAIN_SIZE"))
        ]

        print(
            "\n🔼 "
            + Fore.BLUE
            + f"Just loaded 'raw_data' shaped {raw_data.shape} and 'color_raw_data' shaped{color_raw_data.shape}"
            + Style.RESET_ALL
        )

        return (raw_data, color_raw_data)

    def clean_data(df):
        pass

    def holdout(df):
        pass

    @staticmethod
    def make_directory(directory):
        try:
            os.makedirs(directory)

            print(
                "\n⏹ "
                + Fore.GREEN
                + f"This directory has been created {directory}"
                + Style.RESET_ALL
            )

        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


if __name__ == "__main__":
    pass
