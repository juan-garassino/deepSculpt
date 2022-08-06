from google.cloud import storage
from tensorflow.train import Checkpoint
from deepSculpt.params import BUCKET_NAME, LOCALLY
import os


def upload_checkoint_to_cgp():

    model_checkpoint = "ckpt-" + str(1) + ".index"

    STORAGE_FILENAME = model_checkpoint

    storage_location = f"results/{STORAGE_FILENAME}"

    bucket = storage.Client().bucket(BUCKET_NAME)

    blob = bucket.blob(storage_location)

    blob.upload_from_filename(STORAGE_FILENAME)


def generate_and_save_checkpoint(checkpoint):

    if LOCALLY:

        checkpoint_dir = (
            "/home/juan-garassino/code/juan-garassino/deepSculpt/results/checkpoints"
        )

        checkpoint_prefix = os.path.join(checkpoint_dir, "checkpoint")

        checkpoint.save(file_prefix=checkpoint_prefix)

    if not LOCALLY:

        checkpoint_dir = "deepsculpt/results"

        checkpoint_prefix = os.path.join(checkpoint_dir, "checkpoint")

        checkpoint.save(file_prefix=checkpoint_prefix)

        upload_checkoint_to_cgp()  # , model_checkpoint)# , model_checkpoint)

    return checkpoint


def load_model_from_cgp(checkpoint, manager):

    return checkpoint.restore(manager.latest_checkpoint)

    opt = tf.keras.optimizers.Adam(0.1)
    net = Net()
    dataset = toy_dataset()
    iterator = iter(dataset)
    ckpt = tf.train.Checkpoint(
        step=tf.Variable(1), optimizer=opt, net=net, iterator=iterator
    )
    manager = tf.train.CheckpointManager(ckpt, "./tf_ckpts", max_to_keep=3)

    train_and_checkpoint(net, manager)
