from tensorflow.keras import Sequential, layers
from deepSculpt.params import void_dim, noise_dim

# The generator uses tf.keras.layers.Conv2DTranspose (upsampling) layers to produce an image from a seed (random noise).


def make_three_dimentional_generator():
    model = Sequential()  # Initialize Sequential model
    # The Sequential model is a straight line. You keep adding layers, every new layer takes the output of the previous layer. You cannot make creative graphs with branches
    # The functoinal API Model is completely free to have as many ramifications, inputs and outputs as you need
    model.add(
        layers.Dense(3 * 3 * 3 * 512, use_bias=False, input_shape=(noise_dim,))
    )  # shape 512 noise vector, 7*7*256 flat layer to reshape [7,7,256] | 7 width 7 height 256 channels
    model.add(
        layers.BatchNormalization()
    )  # BatchNormalization doesn't require bias, makes the model faster and more stable
    model.add(layers.ReLU())  # LeakyReLU
    model.add(layers.Reshape((3, 3, 3, 512)))  # reshape [7,7,256]
    assert model.output_shape == (None, 3, 3, 3, 512)  # None is the batch size

    model.add(
        layers.Conv3DTranspose(
            512,
            (3, 3, 3),
            strides=(1, 1, 1),
            padding="same",
            use_bias=False,
            activation="relu",
        )
    )  # 128 Filters... to be the number of channels of the output, (5,5) kernel
    assert model.output_shape == (None, 3, 3, 3, 512)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(
        layers.Conv3DTranspose(
            256,
            (3, 3, 3),
            strides=(2, 2, 2),
            padding="same",
            use_bias=False,
            activation="relu",
        )
    )  # 128 Filters... to be the number of channels of the output, (5,5) kernel
    assert model.output_shape == (None, 6, 6, 6, 256)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(
        layers.Conv3DTranspose(
            128,
            (3, 3, 3),
            strides=(2, 2, 2),
            padding="same",
            use_bias=False,
            activation="relu",
        )
    )  # 128 Filters... to be the number of channels of the output, (5,5) kernel
    assert model.output_shape == (None, 12, 12, 12, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(
        layers.Conv3DTranspose(
            6,
            (3, 3, 3),
            strides=(2, 2, 2),
            padding="same",
            use_bias=False,
            activation="softmax",
        )
    )
    model.add(layers.ThresholdedReLU(theta=0.0))
    model.add(layers.Reshape((24, 24, 24, 6, 1)))
    assert model.output_shape == (None, 24, 24, 24, 6, 1)
    return model


# The discriminator is a CNN-based image classifier it uses tf.keras.layers.Conv2D to classify images as real or fake


def make_three_dimentional_critic():
    model = Sequential()
    model.add(
        layers.Conv3D(
            64,
            (3, 3, 3),
            strides=(2, 2, 2),
            padding="same",
            input_shape=[void_dim, void_dim, void_dim, 6, 1],
        )
    )
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(
        layers.Conv3D(
            128, (3, 3, 3), strides=(2, 2, 2), padding="same", activation="relu"
        )
    )
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(
        layers.Conv3D(
            256, (3, 3, 3), strides=(2, 2, 2), padding="same", activation="relu"
        )
    )
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(
        layers.Conv3D(
            512, (3, 3, 3), strides=(2, 2, 2), padding="same", activation="relu"
        )
    )
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation="sigmoid"))

    return model
