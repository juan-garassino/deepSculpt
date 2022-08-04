from google.cloud import storage
from tensorflow.train import Checkpoint
from deepSculpt.params import BUCKET_NAME


def upload_model_to_cgp(checkpoint):

    STORAGE_FILENAME = checkpoint

    storage_location = f"results/{STORAGE_FILENAME}"

    bucket = storage.Client().bucket(BUCKET_NAME)

    blob = bucket.blob(storage_location)

    blob.upload_from_filename(STORAGE_FILENAME)


def save_model_checkpoint():

    checkpoint = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator,
    )

    upload_model_to_cgp(checkpoint)


def load_model_from_cgp(ckeckpoint):

    STORAGE_FILENAME = checkpoint

    storage_location = f"results/{STORAGE_FILENAME}"

    bucket = storage.Client().bucket(BUCKET_NAME)

    blob = bucket.blob(storage_location)

    blob.download_to_filename(STORAGE_FILENAME)

    checkpoint = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator,
    )

    checkpoint.restore(
        "/content/drive/MyDrive/repositories/deepSculpt/checkpoints/softmax-checkpoints/ckpt-30"
    )

    return checkpoint, checkpoint_prefix
