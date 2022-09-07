from google.cloud import storage
from tensorflow.train import Checkpoint
from deepSculpt.utils.params import BUCKET_NAME, LOCALLY
import os


def upload_checkoint_to_cgp(bucket):

    # model_checkpoint = "ckpt-" + str(1) + ".index"

    # STORAGE_FILENAME = model_checkpoint

    # storage_location = f"results/{STORAGE_FILENAME}"

    # blob = bucket.blob(storage_location)

    # blob.upload_from_filename(STORAGE_FILENAME)

    pass


def generate_and_save_checkpoint(checkpoint, manager, bucket):

    if LOCALLY:

        save_path = manager.save()

        print(
            "Saved checkpoint for step {}: {}".format(int(checkpoint.step), save_path)
        )

        # checkpoint_dir = (
        #    "/home/juan-garassino/code/juan-garassino/deepSculpt/results/checkpoints"
        # )

        # checkpoint_prefix = os.path.join(checkpoint_dir, "checkpoint")

        # checkpoint.save(file_prefix=checkpoint_prefix)

    if not LOCALLY:

        save_path = manager.save()

        print(
            "Saved checkpoint for step {}: {}".format(int(checkpoint.step), save_path)
        )

        # checkpoint_dir = "deepsculpt/results"

        # checkpoint_prefix = os.path.join(checkpoint_dir, "checkpoint")

        # checkpoint.save(file_prefix=checkpoint_prefix)

        # upload_checkoint_to_cgp(bucket)  # , model_checkpoint)# , model_checkpoint)

    return checkpoint, manager


def load_model_from_cgp(checkpoint, manager):

    return checkpoint.restore(manager.latest_checkpoint), manager
