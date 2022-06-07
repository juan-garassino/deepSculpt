from sklearn.preprocessing import LabelEncoder


class Encoder(Sculptor):

    def __init__(self, void):
        self.void = void

    def encoder(colors_array):

        binarizer = LabelEncoder()

        colors = binarizer.fit_transform(colors_array.reshape(-1, 1))

        binary_encoded = np.array([[int(char) for char in "{:03b}".format(color)]
                                for color in colors],
                                dtype=object).reshape((48 * 3, 48, 48))

        classes = binarizer.classes_

        return binary_encoded, classes
