from google.cloud import storage
import matplotlib.pyplot as plt

from deepSculpt.params import BUCKET_NAME, LOCALLY
from deepSculpt.plotter import Plotter


def upload_snapshot_to_gcp(snapshot_name):

    STORAGE_FILENAME = snapshot_name

    storage_location = f"results/{STORAGE_FILENAME}"

    bucket = storage.Client().bucket(BUCKET_NAME)

    blob = bucket.blob(storage_location)

    blob.upload_from_filename(STORAGE_FILENAME)


def generate_and_save_snapshot(model, epoch, preprocessing_class_o, test_input):
    predictions = (
        model(test_input, training=False)  # Notice 'training' is set to False
        .numpy()
        .astype("int")
        .reshape((1, 24, 24, 24, 6))
    )

    o_decoded_volumes, o_decoded_colors = preprocessing_class_o.ohe_decoder(predictions)

    Plotter(
        o_decoded_volumes[0], o_decoded_colors[0], figsize=25, style="#ffffff", dpi=200
    ).plot_sculpture()

    snapshot_name = "image_at_epoch_{:04d}.png".format(epoch)

    plt.savefig(snapshot_name)

    if not LOCALLY:
        upload_snapshot_to_gcp(snapshot_name)
