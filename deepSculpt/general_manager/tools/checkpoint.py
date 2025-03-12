from google.cloud import storage
from tensorflow.train import Checkpoint
import os
from colorama import Fore, Style


def upload_checkoint_to_cgp(bucket):

    # model_checkpoint = "ckpt-" + str(1) + ".index"

    # STORAGE_FILENAME = model_checkpoint

    # storage_location = f"results/{STORAGE_FILENAME}"

    # blob = bucket.blob(storage_location)

    # blob.upload_from_filename(STORAGE_FILENAME)

    pass


def generate_and_save_checkpoint(checkpoint, manager, bucket):

    if int(os.environ.get("INSTANCE")) == 1:

        save_path = manager.save()

        print(
            "\n 🔼 "
            + Fore.BLUE
            + "Saved checkpoint for step {}: {}".format(int(checkpoint.step), save_path)
            + Style.RESET_ALL
        )

    if int(os.environ.get("INSTANCE")) == 0:

        save_path = manager.save()

        print(
            "\n 🔼 "
            + Fore.BLUE
            + "Saved checkpoint for step {}: {}".format(int(checkpoint.step), save_path)
            + Style.RESET_ALL
        )

    return checkpoint, manager


def load_model_from_cgp(checkpoint, manager):

    return checkpoint.restore(manager.latest_checkpoint), manager
