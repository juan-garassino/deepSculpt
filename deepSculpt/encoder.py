from sklearn.preprocessing import LabelEncoder
import numpy as np

class Encoder():

    def __init__(self, sculpture):

        self.sculpture = sculpture
        self.void_dim = self.sculpture.shape[0]

    def encoder(self):

        binarizer = LabelEncoder()

        colors = binarizer.fit_transform(self.sculpture.reshape(-1, 1))

        binary_encoded_sculpture = np.array(
            [[int(char) for char in "{:03b}".format(color)]
             for color in colors],
            dtype=object).reshape(
                (self.void_dim, self.void_dim, self.void_dim, 3, 1))

        classes = binarizer.classes_

        return binary_encoded_sculpture, classes
