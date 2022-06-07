from sklearn.preprocessing import LabelEncoder

class Encoder():

    def __init__(self, sculpture):

        self.sculpture = sculpture

    def encoder(self):

        binarizer = LabelEncoder()

        colors = binarizer.fit_transform(self.sculpture.reshape(-1, 1))

        binary_encoded_sculpture = np.array(
            [[int(char) for char in "{:03b}".format(color)]
             for color in colors],
            dtype=object).reshape((48 * 3, 48, 48))

        classes = binarizer.classes_

        return binary_encoded_sculpture, classes
