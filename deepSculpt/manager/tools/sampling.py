from deepSculpt.curator.curator import Curator
from deepSculpt.manager.manager import Manager
from deepSculpt.curator.tools.preprocessing import OneHotEncoderDecoder
from deepSculpt.curator.tools.params import BUFFER_SIZE
from deepSculpt.manager.tools.plotter import Plotter

import random
import os
from colorama import Fore, Style
import numpy as np
from tensorflow.data import Dataset
import tensorflow as tf


def sampling(
    n_samples=128,
    edge_elements=None,
    plane_elements=None,
    volume_elements=None,
    void_dim=None,
    grid=1,
):  # convert to spare tensor

    # Loads the data
    if int(os.environ.get("CREATE_DATA")) == 0:  # LOADS FROM BIG QUERY

        manager = Manager(
            model_name="",
            data_name="",
            path_colors=os.environ.get("FILE_TO_LOAD_COLORS"),
            path_volumes=os.environ.get("FILE_TO_LOAD_VOLUMES"),
        )

        """curator = Curator(
            path_volumes=os.environ.get("FILE_TO_LOAD_VOLUMES"),
            path_colors=os.environ.get("FILE_TO_LOAD_COLORS"),
        )"""

        if int(os.environ.get("INSTANCE")) == 0:

            path = os.path.join(
                os.environ.get("HOME"), "code", "juan-garassino", "deepSculpt", "data", "sampling"
            )

            volumes, colors = manager.load_locally()

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
                'sampling',
            )

            volumes, colors = manager.load_locally()
            # volumes, colors = manager.load_from_gcp()

        if int(os.environ.get("INSTANCE")) == 2:
            volumes, colors = manager.load_from_query()

        for sample in range(int(os.environ.get('N_SAMPLES_PLOT'))):

            index = random.choices(list(np.arange(0, volumes.shape[0], 1)), k=1)[0]

            Plotter(volumes[index], colors[index], figsize=25, style="#ffffff", dpi=200).plot_sculpture(path)

            print(
            "\n🔽 "
            + Fore.BLUE
            + f"Just ploted 'volume_data[{index}]' and 'material_data[{index}]'"
            + Style.RESET_ALL
        )

    # Creates the data
    if int(os.environ.get("CREATE_DATA")) == 1:  # CREATES AND UPLOADS TO BIG QUERY

        # path
        if int(os.environ.get("INSTANCE")) == 0:
            path = os.path.join(
                os.environ.get("HOME"), "code", "juan-garassino", "deepSculpt", "data"
            )
        # path
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
        # path
        if int(os.environ.get("INSTANCE")) == 2:
            path = os.path.join(
                os.environ.get("HOME"), "code", "juan-garassino", "deepSculpt", "data"
            )

        curator = Curator(  # create=False,
            # locally=True,
            # path_volumes="",
            # path_colors="",
            void_dim=int(void_dim),
            edge_elements=edge_elements,
            plane_elements=plane_elements,
            volume_elements=volume_elements,
            step=None,
            grid=grid,
            directory=path,
            n_samples=int(n_samples),
        )

        # Creates the data
        volumes, colors = curator.create_sculpts(
            # n_edge_elements=n_edge_elements,
            # n_plane_elements=n_plane_elements,
            # n_volume_elements=n_volume_elements,
            # color_edges="dimgrey",
            # color_planes="snow",
            # color_volumes=["crimson", "turquoise", "gold"],
            # verbose=os.environ.get("VERBOSE"),
            # void_dim=int(os.environ.get("VOID_DIM")),
        )

    if isinstance(colors, np.ndarray) == False:
        print("error")

    # Preproccess the data
    preprocessing_class_o = OneHotEncoderDecoder(colors)

    o_encode, o_classes = preprocessing_class_o.ohe_encoder()

    print(
        "\n⏹ "
        + Fore.YELLOW
        + "Just preproccess data from shape {} to {}".format(
            colors.shape, o_encode.shape
        )
        + Style.RESET_ALL
    )

    print(
        "\n⏹ " + Fore.YELLOW + "The classes are: {}".format(o_classes) + Style.RESET_ALL
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


if __name__ == "__main__":
    sampling(
        n_samples=128,
        edge_elements=(1, 0.2, 0.6),
        plane_elements=(1, 0.2, 0.6),
        volume_elements=(1, 0.2, 0.6),
        void_dim=os.environ.get('VOID_DIM'),
        grid=1,
    )
