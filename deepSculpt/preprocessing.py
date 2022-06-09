from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

import numpy as np

class one_hot_encoder_decoder():

    def __init__(self, colors_labels_array, void_dim):

        self.colors_labels_array = colors_labels_array
        self.classes = None
        self.void_dim = void_dim
        self.one_hot_encoder = OneHotEncoder(categories=[[
            'crimson', 'dimgrey', 'gold', 'snow', 'turquoise', None
        ]],
                                             handle_unknown="ignore")

    def ohe_encoder(self):

        colors = self.one_hot_encoder.fit_transform(
            self.colors_labels_array.reshape(-1, 1))

        self.classes = self.one_hot_encoder.categories_

        return colors.toarray().reshape(
            (self.void_dim, self.void_dim, self.void_dim,
             len(self.classes[0]))), self.classes

    def ohe_decoder(self, one_hot_encoded_array):

        decoded_color = self.one_hot_encoder.inverse_transform(
            one_hot_encoded_array.reshape(
                (self.void_dim * self.void_dim * self.void_dim, 6)))

        decoded_void = np.where(decoded_color == None, 0, 1)

        return decoded_void.reshape(
            (self.void_dim, self.void_dim, self.void_dim)), decoded_color.reshape(
                 (self.void_dim, self.void_dim, self.void_dim))

class binary_encoder_decoder():

    def __init__(self, colors_labels_array, void_dim):

        self.colors_labels_array = colors_labels_array
        self.classes = None
        self.void_dim = void_dim

    def binary_encoder(self):

        binarizer = LabelEncoder()

        label_encoded_colors = binarizer.fit_transform(self.colors_labels_array.reshape(-1, 1))

        binary_encoded_colors = np.array( [[int(char) for char in "{:03b}".format(color)] for color in label_encoded_colors] , dtype=object).reshape((self.void_dim,self.void_dim,self.void_dim, 3 ,1))

        self.classes = binarizer.classes_

        return binary_encoded_colors, self.classes

    def binary_decoder(self, binary_encoded_colors):

        dic_color = {'000':self.classes[0],'001':self.classes[1],'010':self.classes[2],'011':self.classes[3],'100':self.classes[4],'101':self.classes[5], "110": None, "111": None}

        dic_void = {'000':1,'001':1,'010':1,'011':1,'100':1,'101':0,"110":0,"111":0}

        decoded_color = np.array([
            dic_color[key] for key in [
                ''.join([str(num) for num in vector]) for vector in [
                    i[0] for i in binary_encoded_colors.reshape(
                        self.void_dim * self.void_dim *
                        self.void_dim, 1, 3).tolist()
                ]
            ]
        ]).reshape((self.void_dim, self.void_dim, self.void_dim))

        decoded_void = np.array([
            dic_void[key] for key in [
                ''.join([str(num) for num in vector]) for vector in [
                    i[0] for i in binary_encoded_colors.reshape(
                        self.void_dim * self.void_dim *
                        self.void_dim, 1, 3).tolist()
                ]
            ]
        ]).reshape((self.void_dim, self.void_dim, self.void_dim))

        return decoded_void, decoded_color
