from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

import numpy as np


class OneHotEncoderDecoder:
    def __init__(self, colors_labels_array):

        self.colors_labels_array = colors_labels_array
        self.void_dim = colors_labels_array.shape[1]
        self.n_sample = colors_labels_array.shape[0]
        self.n_classes = None
        self.classes = None
        self.one_hot_encoder = OneHotEncoder(
            categories=[["crimson", "dimgrey", "gold", "snow", "turquoise", None]],
            handle_unknown="ignore",
        )

    def ohe_encoder(self):

        encode = self.one_hot_encoder.fit_transform(
            self.colors_labels_array.reshape(-1, 1)
        )

        self.classes = self.one_hot_encoder.categories_[0]

        self.n_classes = self.classes.size

        return (
            encode.toarray().reshape(
                (
                    self.n_sample,
                    self.void_dim,
                    self.void_dim,
                    self.void_dim,
                    self.n_classes,
                )
            ),
            self.classes,
        )  # returns encode (samples, shape, shape, shape, classes), classes

    def ohe_decoder(self, one_hot_encoded_array):

        decoded_color = self.one_hot_encoder.inverse_transform(
            one_hot_encoded_array.reshape(
                (
                    self.n_sample * self.void_dim * self.void_dim * self.void_dim,
                    self.n_classes,
                )
            )
        )

        decoded_void = np.where(decoded_color == None, 0, 1)  # where None = 0 else 1

        return decoded_void.reshape(
            (self.n_sample, self.void_dim, self.void_dim, self.void_dim)
        ), decoded_color.reshape(
            (self.n_sample, self.void_dim, self.void_dim, self.void_dim)
        )  # returns volumes(samples, shape, shape, shape) colors(samples, shape, shape, shape)


class BinaryEncoderDecoder:
    def __init__(self, colors_labels_array):

        self.colors_labels_array = colors_labels_array
        self.void_dim = colors_labels_array.shape[1]
        self.n_sample = colors_labels_array.shape[0]
        self.classes = None
        self.n_bit = None
        self.binarizer_encoder = LabelEncoder()

    def binary_encoder(self):

        self.binarizer_encoder = LabelEncoder()

        label_encoded_colors = self.binarizer_encoder.fit_transform(
            self.colors_labels_array.reshape(-1, 1)
        )

        self.classes = self.binarizer_encoder.classes_

        self.n_bit = len(bin(self.classes.size - 1)[2:])

        binary_format = "{:" + f"{self.n_bit}" + "b}"

        binary_encoded_colors = np.array(
            [
                [
                    int(char)
                    for char in binary_format.format(color).replace(
                        " ", "0"
                    )  # crea [0,0,0] desde el string
                ]  # el 03b tiene como output el binario en 3 digitos
                for color in label_encoded_colors  # por cada uno de los flattened samples
            ],
            dtype=object,
        ).reshape(
            (self.n_sample, self.void_dim, self.void_dim, self.void_dim, self.n_bit)
        )

        print(
            f"The output is 'binary_encoded_colors' shaped{binary_encoded_colors.shape} and classes {self.classes}"
        )
        return binary_encoded_colors, self.classes

    def binary_decoder(self, binary_encoded_colors):

        self.n_sample = binary_encoded_colors.shape[0]

        flatten_list = binary_encoded_colors.reshape(
            (self.n_sample * self.void_dim * self.void_dim * self.void_dim, self.n_bit)
        ).tolist()

        decode_preprocess_binary = []

        for pixel in flatten_list:
            decode_preprocess_binary.append("".join(list(str(bit) for bit in pixel)))

        decode_preprocess_decimal = [
            int(encode, base=2) for encode in decode_preprocess_binary
        ]

        decoded_color = self.binarizer_encoder.inverse_transform(
            decode_preprocess_decimal
        ).reshape((self.n_sample, self.void_dim, self.void_dim, self.void_dim))

        decoded_void = np.where(decoded_color == None, 0, 1)

        print(
            f"Just decoded 'decoded_void' shaped {decoded_void.shape} and 'decoded_color' shaped{decoded_color.shape}"
        )

        return (
            decoded_void,
            decoded_color,
            np.unique(decode_preprocess_binary),
            np.unique(decode_preprocess_decimal),
        )
