from tensorflow.keras import Sequential, layers
from deepSculpt.utils.params import VOID_DIM, NOISE_DIM

## GENERATOR


def make_three_dimentional_generator():
    model = Sequential()
    model.add(
        layers.Dense(
            int(VOID_DIM / 8) * int(VOID_DIM / 8) * int(VOID_DIM / 8) * NOISE_DIM,
            use_bias=False,
            input_shape=(NOISE_DIM,),
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(
        layers.Reshape(
            (int(VOID_DIM / 8), int(VOID_DIM / 8), int(VOID_DIM / 8), NOISE_DIM)
        )
    )
    assert model.output_shape == (
        None,
        int(VOID_DIM / 8),
        int(VOID_DIM / 8),
        int(VOID_DIM / 8),
        NOISE_DIM,
    )

    model.add(
        layers.Conv3DTranspose(
            NOISE_DIM,
            (3, 3, 3),
            strides=(1, 1, 1),
            padding="same",
            use_bias=False,
            activation="relu",
        )
    )
    assert model.output_shape == (
        None,
        int(VOID_DIM / 8),
        int(VOID_DIM / 8),
        int(VOID_DIM / 8),
        NOISE_DIM,
    )
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(
        layers.Conv3DTranspose(
            int(NOISE_DIM / 2),
            (3, 3, 3),
            strides=(2, 2, 2),
            padding="same",
            use_bias=False,
            activation="relu",
        )
    )
    assert model.output_shape == (
        None,
        int(VOID_DIM / 4),
        int(VOID_DIM / 4),
        int(VOID_DIM / 4),
        int(NOISE_DIM / 2),
    )
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(
        layers.Conv3DTranspose(
            int(NOISE_DIM / 4),
            (3, 3, 3),
            strides=(2, 2, 2),
            padding="same",
            use_bias=False,
            activation="relu",
        )
    )
    assert model.output_shape == (
        None,
        int(VOID_DIM / 2),
        int(VOID_DIM / 2),
        int(VOID_DIM / 2),
        int(NOISE_DIM / 4),
    )
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
    model.add(layers.Reshape((VOID_DIM, VOID_DIM, VOID_DIM, 6)))
    assert model.output_shape == (None, VOID_DIM, VOID_DIM, VOID_DIM, 6)
    return model


## CRITIC


def make_three_dimentional_critic():
    model = Sequential()
    model.add(
        layers.Conv3D(
            int(NOISE_DIM / 8),
            (3, 3, 3),
            strides=(2, 2, 2),
            padding="same",
            input_shape=[VOID_DIM, VOID_DIM, VOID_DIM, 6],
        )
    )
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(
        layers.Conv3D(
            int(NOISE_DIM / 4),
            (3, 3, 3),
            strides=(2, 2, 2),
            padding="same",
            activation="relu",
        )
    )
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(
        layers.Conv3D(
            int(NOISE_DIM / 2),
            (3, 3, 3),
            strides=(2, 2, 2),
            padding="same",
            activation="relu",
        )
    )
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(
        layers.Conv3D(
            NOISE_DIM, (3, 3, 3), strides=(2, 2, 2), padding="same", activation="relu"
        )
    )
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation="sigmoid"))

    return model
