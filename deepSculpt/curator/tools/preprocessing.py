import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

from colorama import Fore, Style


class OneHotEncoderDecoder:
    """
    This class encodes color and material labels into one-hot encoded arrays and can also decode the one-hot encoded
    arrays back into the original labels.

    Parameters:
    -----------
    colors_labels_array : numpy.ndarray
        A 4D numpy array of shape (n_samples, void_dim, void_dim, void_dim) representing the color labels for each sample.
    materials : list
        A list of strings representing the possible values for the material label.

    Attributes:
    -----------
    colors_labels_array : numpy.ndarray
        The input color labels array.
    void_dim : int
        The dimension of the void (3D space).
    n_samples : int
        The number of samples in the color labels array.
    n_classes : int
        The number of unique values in the materials list.
    classes : numpy.ndarray
        The unique values in the materials list.
    one_hot_encoder : sklearn.preprocessing.OneHotEncoder
        The OneHotEncoder object used to perform one-hot encoding.

    Methods:
    --------
    encode()
        Encodes the color and material labels into one-hot encoded arrays.
    decode(encoded_array)
        Decodes the one-hot encoded arrays back into the original color and material labels.

    Examples:
    ---------
    # Create an instance of OneHotEncoderDecoder
    ohe = OneHotEncoderDecoder(colors_labels_array, materials)

    # Encode the color and material labels
    encoded_array = ohe.encode()

    # Decode the one-hot encoded array
    decoded_labels = ohe.decode(encoded_array)
    """

    def __init__(
        self, colors_labels_array: np.ndarray, materials: list, verbose: int = 1
    ):
        self.materials_labels_array = colors_labels_array
        self.void_dim = self.materials_labels_array.shape[1]
        self.n_samples = self.materials_labels_array.shape[0]
        self.n_classes = None
        self.classes = None
        self.materials = materials
        self.one_hot_encoder = OneHotEncoder(
            categories=[self.materials], handle_unknown="ignore"
        )
        self.verbose = verbose

    def ohe_encode(self):  # -> np.ndarray:
        """
        Encodes the color and material labels into one-hot encoded arrays.

        Returns:
        --------
        encoded_array : numpy.ndarray
            A 5D numpy array of shape (n_samples, void_dim, void_dim, void_dim, n_classes) representing the
            one-hot encoded arrays for each sample and material.

        Raises:
        -------
        ValueError:
            If the input materials list is empty.
        """
        if not self.materials:
            raise ValueError("The list of materials cannot be empty.")

        encoded_array = self.one_hot_encoder.fit_transform(
            self.materials_labels_array.reshape(-1, 1)
        )

        self.classes = self.one_hot_encoder.categories_[0]

        self.n_classes = self.classes.size

        if self.verbose == 1:
            print(
                "\n 🔀 "
                + Fore.RED
                + f"A number of {self.materials_labels_array.shape[0]} samples shaped {self.materials_labels_array.shape} have been encoded in {self.n_classes} classes: {self.classes}"
                + Style.RESET_ALL
            )

        return (
            encoded_array.toarray().reshape(
                (
                    self.n_samples,
                    self.void_dim,
                    self.void_dim,
                    self.void_dim,
                    self.n_classes,
                )
            ),
            self.classes,
        )

    def ohe_decode(self, one_hot_encoded_array):
        """
        Decodes the one-hot encoded arrays back into the original color and material labels.

        Parameters:
        -----------
        encoded_array : numpy.ndarray
            A 5D numpy array of shape (n_samples, void_dim, void_dim, void_dim, n_classes) representing the
            one-hot encoded arrays for each sample and material.

        Returns:
        --------
        decoded_labels : tuple
            A tuple of two numpy arrays. The first array has shape (n_samples, void_dim, void_dim, void_dim) and
            represents the color labels. The second array has shape (n_samples,) and represents the material labels.
        """

        self.n_sample = one_hot_encoded_array.shape[0]

        decoded_color = self.one_hot_encoder.inverse_transform(
            one_hot_encoded_array.reshape(
                (
                    self.n_sample * self.void_dim * self.void_dim * self.void_dim,
                    self.n_classes,
                )
            )
        )

        decoded_void = np.where(decoded_color == None, 0, 1)  # where None = 0 else 1

        if self.verbose == 1:
            print(
                "\n 🔀 "
                + Fore.RED
                + f"A number of {one_hot_encoded_array.shape[0]} samples shaped {one_hot_encoded_array.shape} have been decoded in {self.n_classes} classes: {self.classes}"
                + Style.RESET_ALL
            )

        return decoded_void.reshape(
            (self.n_sample, self.void_dim, self.void_dim, self.void_dim)
        ), decoded_color.reshape(
            (self.n_sample, self.void_dim, self.void_dim, self.void_dim)
        )  # returns volumes(samples, shape, shape, shape) colors(samples, shape, shape, shape)


class BinaryEncoderDecoder:
    def __init__(self, materials_labels_array: np.ndarray, verbose: int = 1):
        """
        A class that encodes and decodes colors into binary format.

        Args:
        colors_labels_array (np.ndarray): A 4D numpy array representing colors in
            label format (integers).

        Attributes:
        colors_labels_array (np.ndarray): The input color array.
        void_dim (int): The shape of the input color array.
        n_sample (int): The number of color samples.
        classes (np.ndarray): The unique color classes.
        n_bit (int): The number of bits required to represent all the color classes.
        binarizer_encoder (LabelEncoder): The label encoder used to encode the colors.

        """
        self.materials_labels_array = materials_labels_array
        self.void_dim = materials_labels_array.shape[1]
        self.n_samples = materials_labels_array.shape[0]
        self.classes = None
        self.n_bit = None
        self.binarizer_encoder = LabelEncoder()
        self.verbose = verbose

    def binary_encode(self):  # -> tuple:
        """
        Encodes the input color array into binary format.

        Returns:
        tuple: A tuple of binary encoded color array and unique color classes.

        """
        label_encoded_colors = self.binarizer_encoder.fit_transform(
            self.materials_labels_array.reshape(-1, 1).flatten()
        )

        self.classes = self.binarizer_encoder.classes_
        self.n_bit = int(
            np.ceil(np.log2(self.classes.size))
        )  # round up to the nearest integer
        binary_format = f"{{:0{self.n_bit}b}}"

        # Convert the label-encoded colors to binary-encoded colors
        binary_encoded_colors = np.array(
            [
                [int(char) for char in binary_format.format(color)]
                for color in label_encoded_colors
            ],
            dtype=object,
        ).reshape(
            (self.n_samples, self.void_dim, self.void_dim, self.void_dim, self.n_bit)
        )

        if self.verbose == 1:
            print(
                "\n 🔀 "
                + Fore.RED
                + f"A number of {self.materials_labels_array.shape[0]} samples shaped {self.materials_labels_array.shape} have been encoded to {binary_encoded_colors.shape} in {len(self.classes)} classes: {self.classes}"
                + Style.RESET_ALL
            )

        return binary_encoded_colors.astype(float), list(self.classes)

    def binary_decode(self, binary_encoded_colors: np.ndarray):  # -> tuple:
        """
        Decodes the input binary-encoded color array into label format.

        Args:
        binary_encoded_colors (np.ndarray): A 5D numpy array representing binary-encoded colors.

        Returns:
        tuple: A tuple of decoded void array, decoded color array, unique binary codes,
            and unique decimal codes.

        """
        self.n_sample = binary_encoded_colors.shape[0]

        flatten_list = (
            binary_encoded_colors.reshape(
                (self.n_samples * self.void_dim**3, self.n_bit)
            )
            .astype(int)
            .tolist()
        )

        decode_preprocess_binary = [
            "".join(str(bit) for bit in pixel) for pixel in flatten_list
        ]
        decode_preprocess_decimal = [
            int(encode, base=2) for encode in decode_preprocess_binary
        ]

        decoded_color = self.binarizer_encoder.inverse_transform(
            decode_preprocess_decimal
        ).reshape((self.n_samples, self.void_dim, self.void_dim, self.void_dim))

        decoded_void = np.where(decoded_color == None, 0, 1)

        if self.verbose == 1:
            print(
                "\n 🔀 "
                + Fore.RED
                + f"A number of {binary_encoded_colors.shape[0]} samples shaped {binary_encoded_colors.shape} have been decoded to {decoded_void.shape} in {len(self.classes)} classes: {self.classes}"
                + Style.RESET_ALL
            )

        return (decoded_void, decoded_color)


class RGBEncoderDecoder:
    pass
