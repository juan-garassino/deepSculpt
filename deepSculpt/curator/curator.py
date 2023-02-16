from deepSculpt.collector.collector import Collector
from deepSculpt.manager.manager import Manager
from deepSculpt.collector.tools.preprocessing import OneHotEncoderDecoder
from deepSculpt.collector.tools.preprocessing import BinaryEncoderDecoder
from deepSculpt.collector.tools.params import BUFFER_SIZE, COLORS
from deepSculpt.manager.tools.plotter import Plotter

import random
import os
from colorama import Fore, Style
import numpy as np
from tensorflow.data import Dataset
import tensorflow as tf


class Curator:  # make manager work with and with out epochs
    def __init__(
        self,
        n_samples=128,
        edge_elements=None,
        plane_elements=None,
        volume_elements=None,
        void_dim=None,
        grid=1,
        binary=1,
    ):
        self.n_samples = n_samples
        self.edge_elements = edge_elements
        self.plane_elements = plane_elements
        self.volume_elements = volume_elements
        self.void_dim = void_dim
        self.grid = grid
        self.binary = binary

    def sampling(self):  # convert to spare tensor

        # Loads the data
        if int(os.environ.get("CREATE_DATA")) == 0:  # LOADS FROM BIG QUERY

            manager = Manager(
                model_name="",
                data_name="",
                path_colors=os.environ.get("FILE_TO_LOAD_COLORS"),
                path_volumes=os.environ.get("FILE_TO_LOAD_VOLUMES"),
            )

            # Local path
            if int(os.environ.get("INSTANCE")) == 0:

                path = os.path.join(
                    os.environ.get("HOME"),
                    "code",
                    "juan-garassino",
                    "deepSculpt",
                    "data",
                    "sampling",
                )

                volumes_void, materials_void = manager.load_locally()

            # Colab path
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

                volumes_void, materials_void = manager.load_locally()
                # volumes_void,  materials_void = manager.load_from_gcp()

            # GCP path
            if int(os.environ.get("INSTANCE")) == 2:
                volumes_void, materials_void = manager.load_from_query()

            for _ in range(int(os.environ.get("N_SAMPLES_PLOT"))):

                index = random.choices(
                    list(np.arange(0, volumes_void.shape[0], 1)), k=1
                )[0]

                Plotter(
                    volumes_void[index],
                    materials_void[index],
                    figsize=25,
                    style="#ffffff",
                    dpi=int(os.environ.get("DPI")),
                ).plot_sculpture(path + f"[{index}]")

                print(
                    "\n 🆗 "
                    + Fore.YELLOW
                    + f"Just ploted 'volume_data[{index}]' and 'material_data[{index}]'"
                    + Style.RESET_ALL
                )

        # Creates the data
        elif (
            int(os.environ.get("CREATE_DATA")) == 1
        ):  # CREATES AND UPLOADS TO BIG QUERY

            # Local path
            if int(os.environ.get("INSTANCE")) == 0:
                path = os.path.join(
                    os.environ.get("HOME"),
                    "code",
                    "juan-garassino",
                    "deepSculpt",
                    "data",
                )

            # Colab path
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
                )

            # GCP path
            if int(os.environ.get("INSTANCE")) == 2:
                path = os.path.join(
                    os.environ.get("HOME"),
                    "code",
                    "juan-garassino",
                    "deepSculpt",
                    "data",
                )

            # Initiates
            curator = Collector(
                void_dim=int(self.void_dim),
                edge_elements=self.edge_elements,
                plane_elements=self.plane_elements,
                volume_elements=self.volume_elements,
                step=None,
                grid=self.grid,
                directory=path,
                n_samples=int(self.n_samples),
            )

            # Creates the data
            volumes_void, materials_void = curator.create_collection()

        # No data
        elif (
            int(os.environ.get("CREATE_DATA")) != 0
            and int(os.environ.get("CREATE_DATA")) != 1
        ):
            print("How do i get data?!")

        else:
            print("Big Error")

        # Returns onehot encoded data
        if self.binary == 0:

            if isinstance(materials_void, np.ndarray) == False:
                print("error")

            materials = [COLORS["edges"], COLORS["planes"]] + COLORS["volumes"] + [None]

            # Preproccess the data
            preprocessing_class_o = OneHotEncoderDecoder(
                materials_void, materials=materials, verbose=1
            )

            o_encode, o_classes = preprocessing_class_o.ohe_encode()

            print(
                "\n 🔀 "
                + Fore.YELLOW
                + "Just preproccess data from shape {} to {}".format(
                    materials_void.shape, o_encode.shape
                )
                + Style.RESET_ALL
            )

            print(
                "\n 🔠 "
                + Fore.YELLOW
                + "The classes are: {}".format(o_classes)
                + Style.RESET_ALL
            )

            # o_encode = tf.sparse.from_dense(o_encode)

            # Creates the dataset
            train_dataset = (
                Dataset.from_tensor_slices(o_encode)
                .shuffle(BUFFER_SIZE)
                .take(int(os.environ.get("TRAIN_SIZE")))
                .batch(int(os.environ.get("BATCH_SIZE")))
            )

            return train_dataset, preprocessing_class_o

        # Returns binary encoded data
        elif self.binary == 1:

            if isinstance(materials_void, np.ndarray) == False:
                print("error")

            # materials = [COLORS["edges"], COLORS["planes"]] + COLORS["volumes"] + [None]

            # Preproccess the data
            preprocessing_class_b = BinaryEncoderDecoder(materials_void)

            b_encode, b_classes = preprocessing_class_b.binary_encode()

            print(
                "\n 🔀 "
                + Fore.YELLOW
                + "Just preproccess data from shape {} to {}".format(
                    materials_void.shape, b_encode.shape
                )
                + Style.RESET_ALL
            )

            print(
                "\n 🔠 "
                + Fore.YELLOW
                + "The classes are: {}".format(b_classes)
                + Style.RESET_ALL
            )

            # o_encode = tf.sparse.from_dense(o_encode)

            # Creates the dataset
            train_dataset = (
                Dataset.from_tensor_slices(b_encode)
                .shuffle(BUFFER_SIZE)
                .take(int(os.environ.get("TRAIN_SIZE")))
                .batch(int(os.environ.get("BATCH_SIZE")))
            )

            return train_dataset, preprocessing_class_b

        # No encoder
        elif self.binary != 0 and self.binary != 1:
            print("broken")

        else:
            print("Big Error")


if __name__ == "__main__":

    curator = Curator(
        n_samples=128,
        edge_elements=(1, 0.2, 0.6),
        plane_elements=(1, 0.2, 0.6),
        volume_elements=(1, 0.2, 0.6),
        void_dim=os.environ.get("VOID_DIM"),
        grid=1,
        binary=1,
    )

    curator.sampling()
