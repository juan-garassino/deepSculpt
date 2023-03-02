from tensorflow.keras import Sequential, layers
import os

## GENERATOR


def tridimensional_simple_generator():
    model = Sequential()
    model.add(
        layers.Dense(
            int(int(os.environ.get("VOID_DIM")) / 8)
            * int(int(os.environ.get("VOID_DIM")) / 8)
            * int(int(os.environ.get("VOID_DIM")) / 8)
            * int(os.environ.get("NOISE_DIM")),
            use_bias=False,
            input_shape=(int(os.environ.get("NOISE_DIM")),),
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(
        layers.Reshape(
            (
                int(int(os.environ.get("VOID_DIM")) / 8),
                int(int(os.environ.get("VOID_DIM")) / 8),
                int(int(os.environ.get("VOID_DIM")) / 8),
                int(os.environ.get("NOISE_DIM")),
            )
        )
    )
    assert model.output_shape == (
        None,
        int(int(os.environ.get("VOID_DIM")) / 8),
        int(int(os.environ.get("VOID_DIM")) / 8),
        int(int(os.environ.get("VOID_DIM")) / 8),
        int(os.environ.get("NOISE_DIM")),
    )

    model.add(
        layers.Conv3DTranspose(
            int(os.environ.get("NOISE_DIM")),
            (3, 3, 3),
            strides=(1, 1, 1),
            padding="same",
            use_bias=False,
            activation="relu",
        )
    )
    assert model.output_shape == (
        None,
        int(int(os.environ.get("VOID_DIM")) / 8),
        int(int(os.environ.get("VOID_DIM")) / 8),
        int(int(os.environ.get("VOID_DIM")) / 8),
        int(os.environ.get("NOISE_DIM")),
    )
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(
        layers.Conv3DTranspose(
            int(int(os.environ.get("NOISE_DIM")) / 2),
            (3, 3, 3),
            strides=(2, 2, 2),
            padding="same",
            use_bias=False,
            activation="relu",
        )
    )
    assert model.output_shape == (
        None,
        int(int(os.environ.get("VOID_DIM")) / 4),
        int(int(os.environ.get("VOID_DIM")) / 4),
        int(int(os.environ.get("VOID_DIM")) / 4),
        int(int(os.environ.get("NOISE_DIM")) / 2),
    )
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(
        layers.Conv3DTranspose(
            int(int(os.environ.get("NOISE_DIM")) / 4),
            (3, 3, 3),
            strides=(2, 2, 2),
            padding="same",
            use_bias=False,
            activation="relu",
        )
    )
    assert model.output_shape == (
        None,
        int(int(os.environ.get("VOID_DIM")) / 2),
        int(int(os.environ.get("VOID_DIM")) / 2),
        int(int(os.environ.get("VOID_DIM")) / 2),
        int(int(os.environ.get("NOISE_DIM")) / 4),
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
    model.add(
        layers.Reshape(
            (
                int(os.environ.get("VOID_DIM")),
                int(os.environ.get("VOID_DIM")),
                int(os.environ.get("VOID_DIM")),
                6,
            )
        )
    )
    assert model.output_shape == (
        None,
        int(os.environ.get("VOID_DIM")),
        int(os.environ.get("VOID_DIM")),
        int(os.environ.get("VOID_DIM")),
        6,
    )
    return model


## CRITIC


def tridimensional_simple_discriminator():
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
        )
    )
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(
        layers.Conv3D(
            int(int(os.environ.get("NOISE_DIM")) / 4),
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
            int(int(os.environ.get("NOISE_DIM")) / 2),
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
            int(os.environ.get("NOISE_DIM")),
            (3, 3, 3),
            strides=(2, 2, 2),
            padding="same",
            activation="relu",
        )
    )
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation="sigmoid"))

    return model
