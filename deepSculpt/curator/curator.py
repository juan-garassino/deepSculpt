from deepSculpt.manager.manager import Manager
from deepSculpt.manager.tools.params import BUFFER_SIZE, COLORS
from deepSculpt.manager.tools.plotter import Plotter

from deepSculpt.collector.collector import Collector

from deepSculpt.curator.tools.preprocessing import (
    OneHotEncoderDecoder,
    BinaryEncoderDecoder,
)

import random
import os
from colorama import Fore, Style
import numpy as np
from tensorflow.data import Dataset
import tensorflow as tf


class Curator:  # make manager work with and with out epochs
    def __init__(
        self,
        processing_method="OHE",
    ):
        self.processing_method = processing_method

    def preprocess_collection_minibatch(
        self, path_volumes, path_colors
    ):  # convert to spare tensor

        manager = Manager()

        # Local path
        if int(os.environ.get("INSTANCE")) == 0:

            path = os.path.join(
                os.environ.get("HOME"), "code", "juan-garassino", "deepSculpt", "data"
            )

            volumes_void, materials_void = manager.load_locally(
                path_volumes, path_colors
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
                "samples",
            )

            # volumes_void,  materials_void = manager.load_from_gcp()
            volumes_void, materials_void = manager.load_locally(
                path_volumes, path_colors
            )

        # Bigquery load
        if int(os.environ.get("INSTANCE")) == 2:
            volumes_void, materials_void = manager.load_from_query()

        # Plot sample from the chunk
        for _ in range(int(os.environ.get("N_SAMPLES_PLOT"))):

            index = random.choices(list(np.arange(0, volumes_void.shape[0], 1)), k=1)[0]

            Plotter(
                figsize=16,
                style="#ffffff",
                dpi=25,
            ).plot_sculpture(
                volumes=volumes_void[index],
                materials=materials_void[index],
                directory=path,  # + f"/sample[{index}]",
                raster_picture=True,
                vector_picture=False,
                volumes_array=False,
                materials_array=False,
                hide_axis=True,
            )

            print(
                "\n 🆗 "
                + Fore.YELLOW
                + f"Just ploted 'volume_data[{index}]' and 'material_data[{index}]'"
                + Style.RESET_ALL
            )

        # Returns onehot encoded data
        if self.processing_method == "OHE":

            if isinstance(materials_void, np.ndarray) == False:
                print("error")

            materials = [COLORS["edges"], COLORS["planes"]] + COLORS["volumes"] + [None]

            # Preproccess the data
            preprocessing_class_o = OneHotEncoderDecoder(materials_void, verbose=1)

            o_encode, o_classes = preprocessing_class_o.ohe_encode()

            print(
                "\n 🔀 "
                + Fore.CYAN
                + "Just preproccess data from shape {} to {}".format(
                    materials_void.shape, o_encode.shape
                )
                + Style.RESET_ALL
            )

            print(
                "\n 🔠 "
                + Fore.CYAN
                + "The classes are: {}".format(o_classes)
                + Style.RESET_ALL
            )

            # o_encode = tf.sparse.from_dense(o_encode)

            # Creates the dataset
            train_dataset = (
                Dataset.from_tensor_slices(o_encode)
                .shuffle(BUFFER_SIZE)
                .take(int(os.environ.get("TRAIN_SIZE")))
                .batch(int(os.environ.get("MINIBATCH_SIZE")))
            )

            return train_dataset, preprocessing_class_o

        # Returns binary encoded data
        elif self.processing_method == "BINARY":

            if isinstance(materials_void, np.ndarray) == False:
                print("error")

            # materials = [COLORS["edges"], COLORS["planes"]] + COLORS["volumes"] + [None]

            # Preproccess the data
            preprocessing_class_b = BinaryEncoderDecoder(materials_void)

            b_encode, b_classes = preprocessing_class_b.binary_encode()

            print(
                "\n 🔀 "
                + Fore.CYAN
                + "Just preproccess data from shape {} to {}".format(
                    materials_void.shape, b_encode.shape
                )
                + Style.RESET_ALL
            )

            print(
                "\n 🔠 "
                + Fore.CYAN
                + "The classes are: {}".format(b_classes)
                + Style.RESET_ALL
            )

            # o_encode = tf.sparse.from_dense(o_encode)

            # Creates the dataset
            train_dataset = (
                Dataset.from_tensor_slices(b_encode)
                .shuffle(BUFFER_SIZE)
                .take(int(os.environ.get("TRAIN_SIZE")))
                .batch(int(os.environ.get("MINIBATCH_SIZE")))
            )

            return train_dataset, preprocessing_class_b

        # Returns RGB encoded data
        elif self.processing_method == "RGB":
            pass

        # No encoder
        elif self.processing_method != "OHE" and self.processing_method != "BINARY":

            print(
                "\n 🆘 "
                + Fore.RED
                + f"The {self.processing_method} processing method is not known"
                + Style.RESET_ALL
            )

        # No encoder
        else:
            print("\n 🆘 " + Fore.RED + "Big Error")


if __name__ == "__main__":

    path_colors = "/home/juan-garassino/code/juan-garassino/deepSculpt/data/material_data[2023-02-28]minibatch[1].npy"

    path_volumes = "/home/juan-garassino/code/juan-garassino/deepSculpt/data/volume_data[2023-02-28]minibatch[1].npy"

    curator = Curator(processing_method="OHE")

    curator.preprocess_collection_minibatch(path_volumes, path_colors)

    curator = Curator(processing_method="BINARY")

    curator.preprocess_collection_minibatch(path_volumes, path_colors)
