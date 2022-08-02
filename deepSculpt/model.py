from tensorflow.keras import Sequential, layers
from deepSculpt.params import VOID_DIM, NOISE_DIM

## GENERATOR

def make_three_dimentional_generator():
    model = Sequential()
    model.add(layers.Dense(3 * 3 * 3 * 512, use_bias=False, input_shape=(NOISE_DIM,)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Reshape((3, 3, 3, 512)))
    assert model.output_shape == (None, 3, 3, 3, 512)

    model.add(
        layers.Conv3DTranspose(
            512,
            (3, 3, 3),
            strides=(1, 1, 1),
            padding="same",
            use_bias=False,
            activation="relu",
        )
    )
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
    )
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
    )
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


## CRITIC

def make_three_dimentional_critic():
    model = Sequential()
    model.add(
        layers.Conv3D(
            64,
            (3, 3, 3),
            strides=(2, 2, 2),
            padding="same",
            input_shape=[VOID_DIM, VOID_DIM, VOID_DIM, 6, 1],
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
