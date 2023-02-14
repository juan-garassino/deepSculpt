import os
import tensorflow as tf
from tensorflow.keras import layers, Sequential

from tensorflow.keras.layers import Input, Dense, Reshape, Concatenate, BatchNormalization, Activation, LeakyReLU, Conv3DTranspose
from tensorflow.keras.models import Model
import os


def make_three_dimensional_generator():
    void_dim = int(os.environ.get("VOID_DIM"))
    noise_dim = int(os.environ.get("NOISE_DIM"))

    inputs = Input(shape=(noise_dim, ))
    x = Dense((void_dim // 8)**3 * noise_dim, use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Reshape((void_dim // 8, void_dim // 8, void_dim // 8, noise_dim))(x)
    assert x.shape == (None, void_dim // 8, void_dim // 8, void_dim // 8,
                       noise_dim)

    skip_connections = []
    filters = [noise_dim, noise_dim // 2, noise_dim // 4, 6]
    strides = [(1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2)]
    for i in range(4):
        if i > 0:
            x = Concatenate()([x, skip_connections[-1]])

        x = Conv3DTranspose(filters[i], (3, 3, 3),
                            strides=strides[i],
                            padding="same",
                            use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        if i != 3:
            skip_connections.append(x)
        else:
            x = Activation('tanh')(x)

    x = Reshape((void_dim, void_dim, void_dim, 6))(x)
    assert x.shape == (None, void_dim, void_dim, void_dim, 6)

    model = Model(inputs=inputs, outputs=x)
    return model




def make_three_dimentional_critic():
    model = Sequential()
    model.add(
        layers.Conv3D(
            int(int(os.environ.get("NOISE_DIM")) / 8),
            (3, 3, 3),
            strides=(2, 2, 2),
            padding="same",
            input_shape=[
                int(os.environ.get("VOID_DIM")),
                int(os.environ.get("VOID_DIM")),
                int(os.environ.get("VOID_DIM")),
                6,
            ],
        ))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(
        layers.Conv3D(
            int(int(os.environ.get("NOISE_DIM")) / 4),
            (3, 3, 3),
            strides=(2, 2, 2),
            padding="same",
            activation="relu",
        ))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(
        layers.Conv3D(
            int(int(os.environ.get("NOISE_DIM")) / 2),
            (3, 3, 3),
            strides=(2, 2, 2),
            padding="same",
            activation="relu",
        ))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(
        layers.Conv3D(
            int(os.environ.get("NOISE_DIM")),
            (3, 3, 3),
            strides=(2, 2, 2),
            padding="same",
            activation="relu",
        ))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation="sigmoid"))

    return model
