#############
## Imports ##
#############

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')


STYLE = "#ffffff"

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import time
from deepSculpt.sculptor import Sculptor
from deepSculpt.params import *
import tensorflow as tf
from tensorflow.keras import Sequential, layers

import keras

import random
import h5py
from scipy.fft import dst, dct, fft
from IPython import display

if load_data:
    os.chdir("/content/drive/MyDrive/data/volumetries")

    raw_data = np.load("raw-data-new.npy")

else:
    os.chdir("/content/drive/MyDrive/data/volumetries")
    raw_data = []
    count = 0

    for sculpture in range(n_samples): #
        count = count + 1
        if count % 10 == 0:
            print("\r{0}".format(count), end='')

        sculptor = Sculptor(void_dim = void_dim,
                      n_edge_elements = 0,
                      n_plane_elements = 8,
                      n_volume_elements = 3,
                      element_edge_min= 16,
                      element_edge_max = 48,
                      element_plane_min = 16,
                      element_plane_max = 30,
                      element_volume_min = 10,
                      element_volume_max = 30,
                      step = 3,
                      verbose = False)

        sculpture = sculptor.generative_sculpt()

        raw_data.append(sculpture)

    raw_data = np.asarray(raw_data).reshape((n_samples, void_dim, void_dim, void_dim, 1))

    np.save("raw-data-new", raw_data, allow_pickle=False)

#####################
## MODEL GENERATOR ##
#####################

def make_three_dimentional_generator():
    model = tf.keras.Sequential() # Initialize Sequential model
    # The Sequential model is a straight line. You keep adding layers, every new layer takes the output of the previous layer. You cannot make creative graphs with branches
    # The functoinal API Model is completely free to have as many ramifications, inputs and outputs as you need
    model.add(layers.Dense(7*7*7*256, use_bias=False, input_shape=(noise_dim,))) # shape 100 noise vector, 7*7*256 flat layer to reshape [7,7,256] | 7 width 7 height 256 channels
    model.add(layers.BatchNormalization()) # BatchNormalization doesn't require bias, makes the model faster and more stable
    model.add(layers.ReLU()) # LeakyReLU
    model.add(layers.Reshape((7, 7, 7, 256))) # reshape [7,7,256]
    assert model.output_shape == (None, 7, 7, 7, 256) # None is the batch size

    model.add(layers.Conv3DTranspose(128, (9, 9, 9), strides=(1, 1, 1), padding='same', use_bias=False)) # 128 Filters... to be the number of channels of the output, (5,5) kernel
    assert model.output_shape == (None, 7, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv3DTranspose(64, (7, 7, 7), strides=(2, 2, 2), padding='same', use_bias=False)) # 128 Filters... to be the number of channels of the output, (5,5) kernel
    assert model.output_shape == (None, 14, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv3DTranspose(32, (7, 7, 7), strides=(2, 2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 28, 28, 28, 32)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv3DTranspose(1, (5, 5, 5), strides=(2, 2, 2), padding='same', use_bias=False, activation='tanh'))
    model.add(layers.ThresholdedReLU(theta = 0))
    assert model.output_shape == (None, 56, 56, 56, 1)
    return model

generator = make_three_dimentional_generator()

##################
## MODEL CRITIC ##
##################

def make_three_dimentional_critic():
    model = tf.keras.Sequential()
    model.add(layers.Conv3D(32, (5, 5, 5), strides=(2, 2, 2), padding='same', input_shape=[56, 56, 56, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv3D(64, (5, 5, 5), strides=(2, 2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv3D(128, (5, 5, 5), strides=(2, 2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))

    return model

discriminator = make_three_dimentional_critic()

decision = discriminator(generator(tf.random.normal([1, noise_dim])))#[0].reshape((28,28,28,1)))

print(decision) # 50 / 50 not trained, will be trained to generate positive values for real pictures and negative for generated ones

###############
## SUMMARIES ##
###############

generator.summary()

discriminator.summary()

##################################
## Model Compile: Loss Function ##
##################################

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True) # we take a BinaryCrossentropy

########################
## Discriminator loss ##
########################

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    # compares the predictions of the discriminator over real images to a matrix of [1s] | must have a tendency/likelihood to 1
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    # compares the predictions of the discriminator over generated images to a matrix of [0s] | must have a tendency/likelihood to 0
    total_loss = real_loss + fake_loss
    return total_loss # Total loss

####################
## Generator loss ##
####################

def generator_loss(fake_output):
    binary_cross_entropy = cross_entropy(tf.ones_like(fake_output), fake_output)
    # the generator's output need to have a tendency to 1, We compare the discriminators decisions on the generated images to an array of [1s]
    return binary_cross_entropy

##############################
## Model Compile: Optimizer ##
##############################

generator_optimizer = tf.keras.optimizers.Adam(1e-4) # SGD INSTEAD???   (Radford et al., 2015)

discriminator_optimizer = tf.keras.optimizers.Adam(1e-4) # SGD INSTEAD???  (Radford et al., 2015)

######################
## Save checkpoints ##
######################

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 #discriminator_optimizer=discriminator_optimizer,
                                 generator=generator) #,
#discriminator=discriminator)

checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 #discriminator_optimizer=discriminator_optimizer,
                                 generator=generator) #,
#discriminator=discriminator)

####################
## Training steps ##
####################

@tf.function

def train_step(images): # train for just ONE STEP aka one forward and back propagation

    noise = tf.random.normal([BATCH_SIZE, noise_dim]) #tf.random.normal([BATCH_SIZE, noise_dim]) # generate the noises [batch size, latent space 100 dimention vector]

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape: # get the gradient for each parameter for this step
        generated_images = generator(noise, training=True) # iterates over the noises

        real_output = discriminator(images, training=True) # trains discriminator based on labeled real pics
        fake_output = discriminator(generated_images, training=True) # trains discriminator based on labeled generated pics
        # why it doesnt traing all at ones

        gen_loss = generator_loss(fake_output) # calculating the generator loss function previously defined
        disc_loss = discriminator_loss(real_output, fake_output) # calculating the descrim loss function previously defined

    print(f"gen loss : {gen_loss}")
    print(f"gen loss : {disc_loss}")

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    # saving the gradients of each trainable variable of the generator
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    # saving the gradients of each trainable variable of the discriminator

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    # applying the gradients on the trainable variables of the generator to update the parameters
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

###################
## Training loop ##
###################

def train(dataset, epochs):

    for epoch in range(epochs):

        start = time.time()

        print("\nStart of epoch %d" % (epoch+1,))
        print("################################")

        for index, image_batch in enumerate(dataset):
            noise = tf.random.normal([BATCH_SIZE, noise_dim]) #tf.random.normal([BATCH_SIZE, noise_dim]) # generate the noises [batch size, latent space 100 dimention vector]

            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape: # get the gradient for each parameter for this step
                generated_images = generator(noise, training=True) # iterates over the noises

                real_output = discriminator(image_batch, training=True) # trains discriminator based on labeled real pics
                fake_output = discriminator(generated_images, training=True) # trains discriminator based on labeled generated pics
                # why it doesnt traing all at ones

                gen_loss = generator_loss(fake_output) # calculating the generator loss function previously defined
                disc_loss = discriminator_loss(real_output, fake_output) # calculating the descrim loss function previously defined

            if (index + 1) % 25 == 0:
                print(f"Minibatch Number {index+1}")
                print(f"The loss of the generator is: {gen_loss}") #, end="\r")
                print(f"The loss of the discriminator is: {disc_loss}") #, end="\r")
                print("################################") #, end="\r")

            gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
            # saving the gradients of each trainable variable of the generator
            gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
            # saving the gradients of each trainable variable of the discriminator

            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
            # applying the gradients on the trainable variables of the generator to update the parameters
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
            # applying the gradients on the trainable variables of the generator to update the parameters

        # Produce images
        display.clear_output(wait=True) # clearing output !!!TO BE CHECKED!!!
        # generate_and_save_images(generator, epoch + 1, seed)

        # Save the model every 15 epochs
        if (epoch + 1) % 5 == 0:
            os.chdir("/content/drive/MyDrive/data/volumetries")
            checkpoint.save(file_prefix = checkpoint_prefix) # saving weights and biases previously calculated by the train step gradients
        if (epoch + 1) % 5 == 0:
            generate_and_save_images(generator, epoch + 1, seed)
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    # Generate after the final epoch
    display.clear_output(wait=True)

predictions = generator(seed, training=False)

predictions.shape

###############################
## Generate and save images  ##
###############################

def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    os.chdir("/content/drive/MyDrive/data/volumetries/images")

    predictions = model(test_input, training=False)

    fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(25, 25), facecolor = (STYLE), subplot_kw=dict(projection="3d"))

    axes.voxels(predictions[0,:,:,:,0], facecolors="orange", edgecolors="k", linewidth=0.05)

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))

"""#TRAINING"""

if load_data:
    os.chdir("/content/drive/MyDrive/data/volumetries")
    train_dataset = tf.data.Dataset.from_tensor_slices(raw_data[:4975]).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
else:
    os.chdir("/content/drive/MyDrive/data/volumetries")
    raw_data = np.load("/content/drive/MyDrive/data/volumetries/raw-data-new.npy")
    train_dataset = tf.data.Dataset.from_tensor_slices(raw_data[:4975]).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

train(train_dataset, EPOCHS)

"""# ITERPOLATIONS & ANIMATIONS"""

checkpoint.restore("/content/drive/MyDrive/data/volumetries/images/training_checkpoints/ckpt-13.index")

noise = tf.random.normal([2, 1, noise_dim]) # Random input vecto [number of samples, Width, Height]

x = noise[0, 0, :].numpy()

y = noise[1, 0, :].numpy()

# uniform interpolation between two points in latent space
def interpolate_points(p1, p2, n_steps=100):
    # interpolate ratios between the points
    ratios = np.linspace(0, 1, num=n_steps)
    # linear interpolate vectors
    vectors = list()
    for ratio in ratios:
        v = (1.0 - ratio) * p1 + ratio * p2
        vectors.append(v)
    return np.asarray(vectors)

interpolation_result = interpolate_points(x, y, n_steps=75)

interpolation_result.shape

os.chdir("/content/drive/MyDrive/data/volumetries/images")

count = 1

for index, restult in enumerate(interpolation_result):
    start = time.time()
    print(f"generating frame number {count}")
    fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(25, 25), facecolor = (STYLE), subplot_kw=dict(projection="3d"))

    for plot in range(1):
        axes.voxels(generator(result[plot].reshape((1,512)), training=False)[0, :, :, :, 0], facecolors="orange", edgecolors="k", linewidth=0.05)
        plt.savefig('image_interpolation_{:04d}.png'.format(count))
    print ('Time for frame {} is {} sec'.format(index + 1, time.time()-start))
    count = count + 1

# plt.plot(interpolation_result[:,0:1], result[:,1:2])

# plt.show

############################
## Display a single image ##
############################

def display_image(epoch_no): # using the epoch number
    return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))

display_image(EPOCHS)

##################
## Animated GIF ##
##################

anim_file = 'dcgan.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob('image_inter*.png')
    filenames = sorted(filenames)
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)

import tensorflow_docs.vis.embed as embed

embed.embed_file(anim_file)

"""# BONUS AND TRASH"""

param_setters = dict()
for var in tf.trainable_variables():
    placeholder = tf.placeholder(var.dtype, var.shape, var.name.split(':')[0]+'_setter')
    param_setters[var.name] = (tf.assign(var, placeholder), placeholder)

generator.load_weights("/content/drive/MyDrive/data/volumetries/training_checkpoints/checkpoint")
