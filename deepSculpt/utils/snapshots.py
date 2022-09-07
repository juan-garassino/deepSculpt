from google.cloud import storage
import matplotlib.pyplot as plt

from deepSculpt.utils.plotter import Plotter
import os
from colorama import Fore, Style


def upload_snapshot_to_gcp(snapshot_name):

    STORAGE_FILENAME = snapshot_name

    storage_location = f"results/{STORAGE_FILENAME}"

    bucket = storage.Client().bucket(os.environ.get("BUCKET_NAME"))

    blob = bucket.blob(storage_location)

    blob.upload_from_filename(STORAGE_FILENAME)

    print(
        "\n🔼 " + Fore.BLUE +
        f"Just uploaded a snapshot to gcp {STORAGE_FILENAME} @ {storage_location}"
        + Style.RESET_ALL)


def generate_and_save_snapshot(model, epoch, preprocessing_class_o, test_input):
    predictions = (
        model(test_input, training=False)  # Notice 'training' is set to False
        .numpy()
        .astype("int")
        .reshape(
            (
                1,
                int(os.environ.get("VOID_DIM")),
                int(os.environ.get("VOID_DIM")),
                int(os.environ.get("VOID_DIM")),
                6,
            )
        )
    )

    o_decoded_volumes, o_decoded_colors = preprocessing_class_o.ohe_decoder(predictions)

    Plotter(
        o_decoded_volumes[0], o_decoded_colors[0], figsize=25, style="#ffffff", dpi=200
    ).plot_sculpture()

    snapshot_name = "image_at_epoch_{:04d}.png".format(epoch)

    plt.savefig(snapshot_name)

    print(
        "\n🔽 " + Fore.BLUE +
        f"Just created a snapshot {snapshot_name} @"
        + Style.RESET_ALL)

    if not os.environ.get("LOCALLY"):
        upload_snapshot_to_gcp(snapshot_name)
