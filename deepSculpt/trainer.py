import matplotlib.pyplot as plt
from IPython import display
import time
import warnings
warnings.filterwarnings("ignore")
import os

from tensorflow.data import Dataset
from tensorflow import GradientTape, function, Variable
from tensorflow.random import normal
from tensorflow.train import Checkpoint, CheckpointManager

from google.cloud import storage
from tensorflow import GradientTape, function

from deepSculpt.preprocessing import OneHotEncoderDecoder
from deepSculpt.data import DataLoaderCreator
from deepSculpt.model import (
    make_three_dimentional_generator,
    make_three_dimentional_critic,
)
from deepSculpt.losses import discriminator_loss, generator_loss
from deepSculpt.optimizers import generator_optimizer, discriminator_optimizer

from deepSculpt.snapshots import generate_and_save_snapshot
from deepSculpt.checkpoint import generate_and_save_checkpoint, load_model_from_cgp

from deepSculpt.params import (
    LOCALLY,
    N_SAMPLES_CREATE,
    VOID_DIM,
    NOISE_DIM,
    BUFFER_SIZE,
    BATCH_SIZE,
    EPOCHS,
    CREATE_DATA,
    BUCKET_NAME,
    SEED,
    MODEL_CHECKPOINT,
    PICTURE_SNAPSHOT,
    TRAIN_SIZE,
    MINIBATCHES,
    FILE_TO_LOAD_VOLUMES,
    FILE_TO_LOAD_COLORS,
)

if CREATE_DATA:

    data = DataLoaderCreator()

    volumes, colors = data.create_sculpts(
        n_samples=N_SAMPLES_CREATE,
        n_edge_elements=0,
        n_plane_elements=2,
        n_volume_elements=2,
        color_edges="dimgrey",
        color_planes="snow",
        color_volumes=["crimson", "turquoise", "gold"],
        verbose=False,
        void_dim=VOID_DIM,
    )

elif not CREATE_DATA:

    data = DataLoaderCreator(
        path_volumes=FILE_TO_LOAD_VOLUMES,
        path_colors=FILE_TO_LOAD_COLORS,
    )

    if LOCALLY:
        volumes, colors = data.load_locally()

    else:
        volumes, colors = data.load_from_gcp()

else:
    print("error")

preprocessing_class_o = OneHotEncoderDecoder(colors)

o_encode, o_classes = preprocessing_class_o.ohe_encoder()

print(f"The classes are: {o_classes}")

train_dataset = (
    Dataset.from_tensor_slices(o_encode)
    .shuffle(BUFFER_SIZE)
    .take(TRAIN_SIZE)
    .batch(BATCH_SIZE)
)

generator = make_three_dimentional_generator()

print(generator.summary())

discriminator = make_three_dimentional_critic()

print(discriminator.summary())

checkpoint_dir = (
    "/home/juan-garassino/code/juan-garassino/deepSculpt/results/checkpoints"
)

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

checkpoint = Checkpoint(
    step=Variable(1),
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    generator=generator,
    discriminator=discriminator,
)

manager = CheckpointManager(
    checkpoint=checkpoint,
    directory=checkpoint_dir,
    max_to_keep=3,
    checkpoint_name="checkpoint",
)

@function  # Notice the use of "tf.function" This annotation causes the function to be "compiled"
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
    # applying the gradients on the trainable variables of the generator to update the parameters

def trainer(dataset, epochs):  # load checkpoint, checkpoint + manager

    load_model_from_cgp(checkpoint, manager)  # REEEEEESTOREEEEEE

    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    for epoch in range(epochs):

        start = time.time()
        print(
            "\n##########################################################################################"
        )
        print(
            "# ----------------------------------------------------------------------------------------"
        )
        print(
            "# -----------------------------------------------------   Start of epoch %d"
            % (epoch + 1,)
        )
        print(
            "# ----------------------------------------------------------------------------------------"
        )
        print(
            "##########################################################################################"
        )

        for index, image_batch in enumerate(dataset):
            noise = normal(
                [BATCH_SIZE, NOISE_DIM]
            )  # tf.random.normal([BATCH_SIZE, noise_dim]) # generate the noises [batch size, latent space 100 dimention vector]

            with GradientTape() as gen_tape, GradientTape() as disc_tape:  # get the gradient for each parameter for this step
                generated_images = generator(
                    noise, training=True
                )  # iterates over the noises

                real_output = discriminator(
                    image_batch, training=True
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

            if (index + 1) % MINIBATCHES[index]:
                minibatch_start = time.time()
                print("\n#============================================================")
                print(f"| Minibatch Number {index+1}")
                print("#============================================================")
                print("|")
                print(f"|  - The loss of the generator is: {gen_loss}")  # , end="\r")
                print(
                    f"|  - The loss of the discriminator is: {disc_loss}"
                )  # , end="\r")
                print("|")
                print(
                    "|  - Time for Minibatch {} is {} sec".format(
                        index + 1, time.time() - minibatch_start
                    )
                )
                print("|")
                print(
                    "#============================================================"
                )  # , end="\r")

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

        if LOCALLY:

            if (epoch + 1) % MODEL_CHECKPOINT == 0:  # Save the model every 15 epochs

                os.chdir("/home/juan-garassino/code/juan-garassino/deepSculpt/results")

                save_path = manager.save()

                print(
                    "Saved checkpoint for step {}: {}".format(
                        int(checkpoint.step), save_path
                    )
                )

                generate_and_save_checkpoint(
                    checkpoint
                )  # saving weights and biases previously calculated by the train step gradients

                checkpoint.step.assign_add(1)

            if (epoch + 1) % PICTURE_SNAPSHOT == 0:
                os.chdir(
                    "/home/juan-garassino/code/juan-garassino/deepSculpt/results/snapshots"
                )
                generate_and_save_snapshot(
                    generator, epoch + 1, preprocessing_class_o, SEED
                )

        if not LOCALLY:
            # Save the model every 15 epochs
            if (epoch + 1) % MODEL_CHECKPOINT == 0:
                generate_and_save_checkpoint(
                    checkpoint
                )  # saving weights and biases previously calculated by the train step gradients
            if (epoch + 1) % PICTURE_SNAPSHOT == 0:
                generate_and_save_snapshot(generator, epoch + 1, SEED)

        print("\n#============================================================")
        print("|  - Time for epoch {} is {} sec".format(epoch + 1, time.time() - start))
        print("#============================================================")

        plt.close("all")

    # Generate after the final epoch
    display.clear_output(wait=True)
    # generate_and_save_images(generator, epochs, seed)

if __name__ == "__main__":
    trainer(train_dataset, EPOCHS)
