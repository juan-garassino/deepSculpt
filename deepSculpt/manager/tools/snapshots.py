from google.cloud import storage
import matplotlib.pyplot as plt

from deepSculpt.manager.tools.plotter import Plotter
from deepSculpt.manager.manager import Manager

from datetime import datetime
import os
from colorama import Fore, Style


def upload_snapshot_to_gcp(snapshot_name):

    STORAGE_FILENAME = snapshot_name

    storage_location = f"results/{STORAGE_FILENAME}"

    bucket = storage.Client().bucket(os.environ.get("BUCKET_NAME"))

    blob = bucket.blob(storage_location)

    blob.upload_from_filename(STORAGE_FILENAME)

    print(
        "\n 🔼 "
        + Fore.BLUE
        + f"Just uploaded a snapshot to gcp {STORAGE_FILENAME} @ {storage_location}"
        + Style.RESET_ALL
    )


def generate_and_save_snapshot(
    model, epoch, preprocessing_class_o, snapshot_input, directory
):

    # Generates the sculpture
    predictions = (
        model(snapshot_input, training=False)  # Notice 'training' is set to False
        .numpy()
        .astype("int")
        .reshape(
            (
                int(os.environ.get("SCULPTS_GEN")),
                int(os.environ.get("VOID_DIM")),
                int(os.environ.get("VOID_DIM")),
                int(os.environ.get("VOID_DIM")),
                6,
            )
        )
    )

    # Decodes the structure to be plotted
    o_decoded_volumes, o_decoded_colors = preprocessing_class_o.ohe_decode(predictions)

    for prediction in range(int(os.environ.get("SCULPTS_GEN"))):
        # Plots the Sculpture
        Plotter(
            o_decoded_volumes[prediction],
            o_decoded_colors[prediction],
            figsize=25,
            style="#ffffff",
            dpi=int(os.environ.get("DPI")),
        ).plot_sculpture(directory + f"[{prediction}]", raster_picture=True, volumes_array= True, hide_axis=True)
