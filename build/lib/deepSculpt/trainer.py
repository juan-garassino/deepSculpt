from tensorflow.data import Dataset
from tensorflow import GradientTape
from tensorflow.random import normal

from IPython import display
import time
from deepSculpt.params import (
    LOCALLY,
    N_SAMPLES,
    VOID_DIM,
    NOISE_DIM,
    BUFFER_SIZE,
    BATCH_SIZE,
    EPOCHS,
    CREATE_DATA,
)
from deepSculpt.preprocessing import OneHotEncoderDecoder
from deepSculpt.data import DataLoaderCreator
from deepSculpt.model import (
    make_three_dimentional_generator,
    make_three_dimentional_critic,
)

from deepSculpt.losses import discriminator_loss, generator_loss
from deepSculpt.optimizers import generator_optimizer, discriminator_optimizer

if CREATE_DATA:

    data = DataLoaderCreator()

    volumes, colors = data.create_sculpts(
        n_samples=N_SAMPLES,
        n_edge_elements=0,
        n_plane_elements=2,
        n_volume_elements=2,
        color_edges="dimgrey",
        color_planes="snow",
        color_volumes=["crimson", "turquoise", "gold"],
        verbose=False,
        void_dim=VOID_DIM,
    )

else:

    data = DataLoaderCreator(
        path_volumes="raw-data[2022-07-26].npy",
        path_colors="color-raw-data[2022-07-26].npy",
    )
    if LOCALLY:
        volumes, colors = data.load_locally()

    else:
        volumes, colors = data.load_from_gcp()

## PREPRO

preprocessing_class_o = OneHotEncoderDecoder(colors)

o_encode, o_classes = preprocessing_class_o.ohe_encoder()

train_dataset = (
    Dataset.from_tensor_slices(o_encode).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
)

generator = make_three_dimentional_generator()

discriminator = make_three_dimentional_critic()

from tensorflow import GradientTape, function

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@function

def train_step(images):  # train for just ONE STEP aka one forward and back propagation

    noise = normal(
        [BATCH_SIZE, noise_dim]
    )  # tf.random.normal([BATCH_SIZE, noise_dim]) # generate the noises [batch size, latent space 100 dimention vector]

    with GradientTape() as gen_tape, GradientTape() as disc_tape:  # get the gradient for each parameter for this step
        generated_images = generator(noise, training=True)  # iterates over the noises

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

    print(f"gen loss : {gen_loss}")
    print(f"gen loss : {disc_loss}")

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

def train(dataset, epochs):

    # load checkpoint

    for epoch in range(epochs):

        start = time.time()

        print("\nStart of epoch %d" % (epoch + 1,))
        print("################################")

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

            if (index + 1) % 25 == 0:
                print(f"Minibatch Number {index+1}")
                print(f"The loss of the generator is: {gen_loss}")  # , end="\r")
                print(f"The loss of the discriminator is: {disc_loss}")  # , end="\r")
                print("################################")  # , end="\r")

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

        # Save the model every 15 epochs
        if (epoch + 1) % 20 == 0:
            os.chdir("/content/drive/MyDrive/repositories/deepSculpt/checkpoints")
            checkpoint.save(
                file_prefix=checkpoint_prefix
            )  # saving weights and biases previously calculated by the train step gradients
        if (epoch + 1) % 2 == 0:
            generate_and_save_images(generator, epoch + 1, seed)
        print("Time for epoch {} is {} sec".format(epoch + 1, time.time() - start))

        plt.close("all")

    # Generate after the final epoch
    display.clear_output(wait=True)
    # generate_and_save_images(generator, epochs, seed)


if __name__ == "__main__":
    train(train_dataset, EPOCHS)
