import matplotlib.pyplot as plt
from IPython import display
import time
import warnings
import re

warnings.filterwarnings("ignore")

from colorama import Fore, Style
import os
import numpy as np
from google.cloud import storage

from tensorflow.data import Dataset
from tensorflow import GradientTape, function, Variable
from tensorflow.random import normal
from tensorflow.train import Checkpoint, CheckpointManager

from deepSculpt.general_manager.manager import Manager
from deepSculpt.network_trainer.tools.losses import discriminator_loss, generator_loss

from deepSculpt.network_trainer.tools.simple_model import (
    tridimensional_simple_discriminator,
    tridimensional_simple_generator,
)

from deepSculpt.network_trainer.tools.complex_model import (
    tridimensional_complex_discriminator,
    tridimensional_complex_generator,
)

from deepSculpt.network_trainer.tools.skip_model import (
    tridimensional_skip_connection_discriminator,
    tridimensional_skip_connection_generator,
)

from deepSculpt.network_trainer.tools.monochrome_model import (
    tridimensional_monochrome_discriminator,
    tridimensional_monochrome_generator,
)

from deepSculpt.network_trainer.tools.optimizers import (
    generator_optimizer,
    discriminator_optimizer,
)
from deepSculpt.general_manager.tools.snapshots import (
    generate_and_save_snapshot,
    upload_snapshot_to_gcp,
)
from deepSculpt.general_manager.tools.checkpoint import (
    generate_and_save_checkpoint,
    load_model_from_cgp,
)
from deepSculpt.general_manager.tools.params import SEED, MINIBATCHES
from deepSculpt.dataset_generator.curator import Curator

if os.environ.get("COLOR") == 0:  # MONOCHROME
    pass

if int(os.environ.get("COLOR")) == 1:  # COLOR

    # Loads and process data
    curator = Curator(processing_method="OHE")

    # Initiates the Generator

    generator = tridimensional_skip_connection_generator()

    import tensorflow as tf

    def add_regularization(model):
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Conv3DTranspose):
                dropout_layer = tf.keras.Sequential([tf.keras.layers.Dropout(0.3)])
                layer = tf.keras.Sequential([layer, dropout_layer])
        return model

    generator = add_regularization(generator)

    generator.compile()

    print("\n ❎ " + Fore.RED + "The Generators summary is" + Fore.YELLOW + "\n ")

    print(generator.summary())

    # Initiates the Discriminator

    discriminator = tridimensional_skip_connection_discriminator()

    discriminator.compile()

    print("\n ❎ " + Fore.RED + "The Discriminators summary is" + Fore.YELLOW + "\n ")

    print(discriminator.summary())

    print(Style.RESET_ALL)

    ## Path local instance enviroment on COMPUTER

    if int(os.environ.get("INSTANCE")) == 0:

        checkpoint_dir = os.path.join(
            os.environ.get("HOME"),
            "code",
            "juan-garassino",
            "deepSculpt",
            "results",
            "checkpoints",
        )

        Manager.make_directory(checkpoint_dir)

    ## Path local instance eviroment on COLAB

    if int(os.environ.get("INSTANCE")) == 1:

        checkpoint_dir = os.path.join(
            os.environ.get("HOME"),
            "..",
            "content",
            "drive",
            "MyDrive",
            "repositories",
            "deepSculpt",
            "results",
            "checkpoints",
        )

        Manager.make_directory(checkpoint_dir)

    ## Path cloud instance enviroment in GCP

    if int(os.environ.get("INSTANCE")) == 2:

        checkpoint_dir = "gs://deepsculpt/checkpoints"

        bucket = storage.Client().bucket(os.environ.get("BUCKET_NAME"))

    # Initiates a Checkpoint Object

    checkpoint = Checkpoint(
        step=Variable(1),
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator,
    )

    # Initiates a Checkpoint Manager Object

    manager = CheckpointManager(
        checkpoint=checkpoint,
        directory=checkpoint_dir,
        max_to_keep=3,
        checkpoint_name="checkpoint",
    )


"""@function  # Notice the use of "tf.function" This annotation causes the function to be "compiled"
def train_step(images):  # train for just ONE STEP aka one forward and back propagation

    with GradientTape() as gen_tape, GradientTape() as disc_tape:  # get the gradient for each parameter for this step
        generated_images = generator(SEED, training=True)  # iterates over the noises

        real_output = discriminator(
            images, training=True
        )  # trains discriminator based on labeled real pics
        fake_output = discriminator(
            generated_images, training=True
        )  # trains discriminator based on labeled generated pics
        # why it doesnt traing all at ones

        gen_loss = generator_loss(
            fake_output
        )  # calculating the generator loss function previously defined
        disc_loss = discriminator_loss(
            real_output, fake_output
        )  # calculating the descrim loss function previously defined

    # print(f"gen loss : {gen_loss}")
    # print(f"gen loss : {disc_loss}")

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    # saving the gradients of each trainable variable of the generator

    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables
    )
    # saving the gradients of each trainable variable of the discriminator

    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, generator.trainable_variables)
    )
    # applying the gradients on the trainable variables of the generator to update the parameters
    discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator, discriminator.trainable_variables)
    )
    # applying the gradients on the trainable variables of the generator to update the parameters"""


"""@function
def train_step(images, gen_steps=1, disc_steps=1):
    for i in range(gen_steps):
        with GradientTape() as gen_tape:
            generated_images = generator(SEED, training=True)
            fake_output = discriminator(generated_images, training=True)
            gen_loss = generator_loss(fake_output)
        gradients_of_generator = gen_tape.gradient(
            gen_loss, generator.trainable_variables
        )
        generator_optimizer.apply_gradients(
            zip(gradients_of_generator, generator.trainable_variables)
        )
    for i in range(disc_steps):
        with GradientTape() as disc_tape:
            real_output = discriminator(images, training=True)
            fake_output = discriminator(generated_images, training=True)
            disc_loss = discriminator_loss(real_output, fake_output)
        gradients_of_discriminator = disc_tape.gradient(
            disc_loss, discriminator.trainable_variables
        )
        discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, discriminator.trainable_variables)
        )"""


def trainer(
    collection_folder,
    data_date,
    # n_minibatches,
    n_epochs,
):  # load checkpoint, checkpoint + manager

    if int(os.environ.get("INSTANCE")) == 2:
        load_model_from_cgp(checkpoint, manager)

    if manager.latest_checkpoint:
        print(
            "\n 🔽 "
            + Fore.YELLOW
            + "Restored from {}...".format(manager.latest_checkpoint)
            + Style.RESET_ALL
        )
    else:
        print("\n ✅ " + Fore.GREEN + "Initializing from scratch" + Style.RESET_ALL)

    for epoch in range(n_epochs):

        start = time.time()

        print("\n ⏩ " + Fore.RED + "Epoch number %d" % (epoch + 1,) + Style.RESET_ALL)

        chunk_files = [
            f
            for f in os.listdir(collection_folder)
            if os.path.isfile(os.path.join(collection_folder, f))
        ]

        for index in range(len(chunk_files) // 2):

            print(
                "\n ⏩ "
                + Fore.RED
                + "Chunk number %d Epoch number %d" % (index + 1, epoch + 1)
                + Style.RESET_ALL
            )

            if int(os.environ.get("INSTANCE")) == 0:

                root_path = os.path.join(
                    os.environ.get("HOME"),
                    "code",
                    "juan-garassino",
                    "deepSculpt",
                    "data",
                )

                path_volumes = (
                    f"{root_path}/volume_data[{data_date}]chunk[{index + 1}].npy"
                )

                path_materials = (
                    f"{root_path}/material_data[{data_date}]chunk[{index + 1}].npy"
                )

            if int(os.environ.get("INSTANCE")) == 1:

                root_path = os.path.join(
                    os.environ.get("HOME"),
                    "..",
                    "content",
                    "drive",
                    "MyDrive",
                    "repositories",
                    "deepSculpt",
                    "data",
                )

                path_volumes = (
                    f"{root_path}/volume_data[{data_date}]chunk[{index + 1}].npy"
                )

                path_materials = (
                    f"{root_path}/material_data[{data_date}]chunk[{index + 1}].npy"
                )

            print(
                "\n ✅ " + Fore.GREEN + f"Volumes Path: {path_volumes}" + Style.RESET_ALL
            )

            print(
                "\n ✅ "
                + Fore.GREEN
                + f"Materials Path: {path_materials}"
                + Style.RESET_ALL
            )

            (
                chunk_sculpts,
                preprocessing_class_o,
            ) = curator.preprocess_collection_minibatch(path_volumes, path_materials)

            for index, sculpts_batch in enumerate(chunk_sculpts):

                noise = normal(
                    [
                        int(os.environ.get("MINIBATCH_SIZE")),
                        int(os.environ.get("NOISE_DIM")),
                    ]
                )

                with GradientTape() as gen_tape, GradientTape() as disc_tape:  # get the gradient for each parameter for this step
                    generated_images = generator(
                        noise, training=True
                    )  # iterates over the noises

                    real_output = discriminator(
                        sculpts_batch, training=True
                    )  # trains discriminator based on labeled real pics
                    fake_output = discriminator(
                        generated_images, training=True
                    )  # trains discriminator based on labeled generated pics
                    # why it doesnt traing all at ones

                    gen_loss = generator_loss(
                        fake_output
                    )  # calculating the generator loss function previously defined
                    disc_loss = discriminator_loss(
                        real_output, fake_output
                    )  # calculating the descrim loss function previously defined

                print(
                    "\n ⏩ "
                    + Fore.RED
                    + f"Iteartion number {index} of minibatches [ {' | '.join([num.zfill(3) for num in [str(batch) for batch in MINIBATCHES]])} ]"
                    + Style.RESET_ALL
                )

                if (index + 1) % MINIBATCHES[index]:
                    minibatch_start = time.time()

                    print(
                        "\n ⏩ "
                        + Fore.MAGENTA
                        + f"Minibatch number {index + 1} epoch {epoch + 1}"
                        + Style.RESET_ALL
                    )

                    print(
                        "\n 📶 "
                        + Fore.CYAN
                        + "Discriminator Loss: {:.4f}, Generator Loss: {:.4f}".format(
                            disc_loss, gen_loss
                        )
                        + Style.RESET_ALL
                    )

                    print(
                        "\n 📶 "
                        + Fore.MAGENTA
                        + "Time for minibatches between {} and {} is {} sec".format(
                            (index * int(os.environ.get("MINIBATCH_SIZE"))),
                            ((index + 1) * int(os.environ.get("MINIBATCH_SIZE"))),
                            time.time() - minibatch_start,
                        )
                        + Style.RESET_ALL
                    )

                gradients_of_generator = gen_tape.gradient(
                    gen_loss, generator.trainable_variables
                )
                # saving the gradients of each trainable variable of the generator
                gradients_of_discriminator = disc_tape.gradient(
                    disc_loss, discriminator.trainable_variables
                )
                # saving the gradients of each trainable variable of the discriminator

                generator_optimizer.apply_gradients(
                    zip(gradients_of_generator, generator.trainable_variables)
                )
                # applying the gradients on the trainable variables of the generator to update the parameters
                discriminator_optimizer.apply_gradients(
                    zip(gradients_of_discriminator, discriminator.trainable_variables)
                )
                # applying the gradients on the trainable variables of the generator to update the parameters

        # Produce images
        display.clear_output(wait=True)  # clearing output !!!TO BE CHECKED!!!
        # generate_and_save_images(generator, epoch + 1, seed)

        # Saves checkpoint and snapshots locally
        if int(os.environ.get("INSTANCE")) == 0:

            if (epoch + 1) % int(os.environ.get("MODEL_CHECKPOINT")) == 0:

                out_dir = os.path.join(
                    os.environ.get("HOME"),
                    "code",
                    "juan-garassino",
                    "deepSculpt",
                    "results",
                    "checkpoints",
                )

                save_path = manager.save()

                print(
                    "\n 🔼 "
                    + Fore.BLUE
                    + "Saved checkpoint for step {}: {}".format(
                        int(checkpoint.step), save_path
                    )
                    + Style.RESET_ALL
                )

                checkpoint.step.assign_add(1)

                out_dir = os.path.join(
                    os.environ.get("HOME"),
                    "code",
                    "juan-garassino",
                    "deepSculpt",
                    "results",
                    "network",
                )

                generator.save(out_dir)

                metrics = {}

                params = {}

                # Save the checkpoint mlfow
                Manager.save_mlflow_model(metrics=metrics, params=params, model=None)

                print(
                    "\n 🔼 "
                    + Fore.BLUE
                    + "Saved checkpoint for step {}: mlfow".format(int(checkpoint.step))
                    + Style.RESET_ALL
                )

            if (epoch + 1) % int(os.environ.get("PICTURE_SNAPSHOT")) == 0:

                out_dir = os.path.join(
                    os.environ.get("HOME"),
                    "code",
                    "juan-garassino",
                    "deepSculpt",
                    "results",
                    "snapshots",
                )

                generate_and_save_snapshot(
                    generator, epoch + 1, preprocessing_class_o, SEED, out_dir
                )

        # Saves Checkpoint and snapshot to COLAB
        if int(os.environ.get("INSTANCE")) == 1:

            # Save the checkpoint
            if (epoch + 1) % int(os.environ.get("MODEL_CHECKPOINT")) == 0:

                out_dir = os.path.join(
                    os.environ.get("HOME"),
                    "..",
                    "content",
                    "drive",
                    "MyDrive",
                    "repositories",
                    "deepSculpt",
                    "results",
                    "checkpoints",
                )

                save_path = manager.save()

                print(
                    "\n 🔼 "
                    + Fore.BLUE
                    + "Saved checkpoint for step {}: {}".format(
                        int(checkpoint.step), save_path
                    )
                    + Style.RESET_ALL
                )

                checkpoint.step.assign_add(1)

                out_dir = os.path.join(
                    os.environ.get("HOME"),
                    "..",
                    "content",
                    "drive",
                    "MyDrive",
                    "repositories",
                    "deepSculpt",
                    "results",
                    "network",
                )

                generator.save(out_dir)

                metrics = {}

                params = {}

                # Save the checkpoint mlfow
                Manager.save_mlflow_model(metrics=metrics, params=params, model=None)

                print(
                    "\n 🔼 "
                    + Fore.BLUE
                    + "Saved checkpoint for step {}: mlfow".format(int(checkpoint.step))
                    + Style.RESET_ALL
                )

            if (epoch + 1) % int(os.environ.get("PICTURE_SNAPSHOT")) == 0:

                out_dir = os.path.join(
                    os.environ.get("HOME"),
                    "..",
                    "content",
                    "drive",
                    "MyDrive",
                    "repositories",
                    "deepSculpt",
                    "results",
                    "snapshots",
                )

                generate_and_save_snapshot(
                    generator,
                    epoch + 1,
                    preprocessing_class_o,
                    SEED,
                    out_dir,
                    hide_axis=False,
                )

                out_dir = os.path.join(
                    os.environ.get("HOME"),
                    "..",
                    "content",
                    "results",
                    "deepSculpt",
                    "predictions",
                )

                Manager.make_directory(out_dir)

                generate_and_save_snapshot(
                    generator, epoch + 1, preprocessing_class_o, SEED, out_dir
                )

                snapshot_name = "{}/image_at_epoch_{:04d}.png".format(out_dir, epoch)

                plt.savefig(snapshot_name)

                print(
                    "\n 🔽 "
                    + Fore.BLUE
                    + f"Just created a snapshot {snapshot_name.split('/')[-1]} @ {out_dir}"
                    + Style.RESET_ALL
                )

                if int(os.environ.get("INSTANCE")) == 0:
                    upload_snapshot_to_gcp(snapshot_name)

        # Saves checkpoint and snapshots to GCP
        if int(os.environ.get("INSTANCE")) == 2:
            # Save the model every 15 epochs
            if (epoch + 1) % int(os.environ.get("MODEL_CHECKPOINT")) == 0:
                generate_and_save_checkpoint(
                    checkpoint, manager, bucket
                )  # saving weights and biases previously calculated by the train step gradients
            if (epoch + 1) % int(os.environ.get("PICTURE_SNAPSHOT")) == 0:
                generate_and_save_snapshot(
                    generator, epoch + 1, preprocessing_class_o, SEED
                )

        # Saves checkpoint and snapshots to MLFOW
        if int(os.environ.get("INSTANCE")) == 2:
            print("to MLFLOW")

        print(
            "\n 📶 "
            + Fore.MAGENTA
            + "Time for epoch {} is {} sec".format(epoch + 1, time.time() - start)
            + Style.RESET_ALL
        )

        plt.close("all")

    # Generate after the final epoch
    display.clear_output(wait=True)
    # generate_and_save_images(generator, epochs, seed)


if __name__ == "__main__":

    if int(os.environ.get("INSTANCE")) == 0:

        collection_folder = os.path.join(
            os.environ.get("HOME"), "code", "juan-garassino", "deepSculpt", "data"
        )

    if int(os.environ.get("INSTANCE")) == 1:
        collection_folder = os.path.join(
            os.environ.get("HOME"),
            "..",
            "content",
            "drive",
            "MyDrive",
            "repositories",
            "deepSculpt",
            "data",
        )

    curator = Curator(processing_method="OHE")

    template = [
        f
        for f in os.listdir(collection_folder)
        if os.path.isfile(os.path.join(collection_folder, f))
    ][0]

    date = re.search(r"\[(\d{4}-\d{2}-\d{2})\]", template).group(1)

    trainer(
        collection_folder,
        date,
        # curator,
        int(os.environ.get("EPOCHS")),
    )
