from tensorflow.data import Dataset
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow import ones_like, zeros_like, GradientTape, function
from tensorflow.random import normal
from IPython import display

import time
from deepSculpt.params import (
    load_data,
    n_samples,
    void_dim,
    BUFFER_SIZE,
    BATCH_SIZE,
    EPOCHS,
    noise_dim,
    num_examples_to_generate,
    seed,
    BUCKET_NAME,
    BUCKET_TRAIN_DATA_PATH,
    MODEL_BASE_PATH,
)
from deepSculpt.preprocessing import OneHotEncoderDecoder
from deepSculpt.data import DataLoaderCreator
from deepSculpt.model import (
    make_three_dimentional_generator,
    make_three_dimentional_critic,
)

data = DataLoaderCreator(
    path_volumes="raw-data[2022-07-26].npy",
    path_colors="color-raw-data[2022-07-26].npy",
)

volumes, colors = data.get_data()

preprocessing_class_o = OneHotEncoderDecoder(colors)

o_encode, o_classes = preprocessing_class_o.ohe_encoder()

train_dataset = (
    Dataset.from_tensor_slices(o_encode).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
)

generator = make_three_dimentional_generator()

discriminator = make_three_dimentional_critic()

##################################
## Model Compile: Loss Function ##
##################################

cross_entropy = BinaryCrossentropy(from_logits=True)  # we take a BinaryCrossentropy

# Binary cross entropy compares each of the predicted probabilities to actual class output which can be either 0 or 1
# Binary Cross Entropy is the negative average of the log of corrected predicted probabilities

########################
## Discriminator loss ## ## check if the loss is calculating with the 6 channel array or not!!!
########################

# quantifies how well the discriminator is able to distinguish real images from generated


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(ones_like(real_output), real_output)
    # compares the predictions of the discriminator over real images to a matrix of [1s] | must have a tendency/likelihood to 1
    fake_loss = cross_entropy(zeros_like(fake_output), fake_output)
    # compares the predictions of the discriminator over generated images to a matrix of [0s] | must have a tendency/likelihood to 0
    total_loss = real_loss + fake_loss
    return total_loss  # Total loss


####################
## Generator loss ##
####################

# quantifies how well it was able to trick the discriminator, if the generator is performing well, the discriminator will classify the fake images as real (1).


def generator_loss(fake_output):
    binary_cross_entropy = cross_entropy(ones_like(fake_output), fake_output)
    # the generator's output need to have a tendency to 1, We compare the discriminators decisions on the generated images to an array of [1s]
    return binary_cross_entropy


##############################
## Model Compile: Optimizer ##
##############################

# Two different optimizers since we train two separate networks:

generator_optimizer = Adam(1e-3)  # SGD INSTEAD???   (Radford et al., 2015)

discriminator_optimizer = Adam(1e-4)  # SGD INSTEAD???  (Radford et al., 2015)

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@function

####################
## Training steps ##
####################


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


###################
## Training loop ##
###################

# training loop itself using train_step function previously defined


def train(dataset, epochs):

    # load checkpoint

    for epoch in range(epochs):

        start = time.time()

        print("\nStart of epoch %d" % (epoch + 1,))
        print("################################")

        for index, image_batch in enumerate(dataset):
            noise = normal(
                [BATCH_SIZE, noise_dim]
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
