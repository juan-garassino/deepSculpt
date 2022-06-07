class Decoder():

    def __init__(self, binary_array_sculpture, classes, void_dim):

        self.binary_array_sculpture = binary_array_sculpture
        self.classes = classes
        self.void_dim = void_dim

    def decoder(self):

        dic_color = {
            '000': self.classes[0],
            '001': self.classes[1],
            '010': self.classes[2],
            '011': self.classes[3],
            '100': self.classes[4],
            '101': self.classes[5],
            "110": self.classes[6],
            "111": self.classes[7]
        }

        dic_void = {'000':1,'001':1,'010':1,'011':1,'100':1,'101':0,"110":0,"111":0}

        decoded_color = np.array([
            dic_color[key] for key in [
                ''.join([str(num) for num in vector]) for vector in [
                    i[0] for i in self.binary_array_sculpture.reshape(
                        self.void_dim * self.void_dim *
                        self.void_dim, 1, 3).tolist()
                ]
            ]
        ]).reshape((self.void_dim, self.void_dim, self.void_dim))

        decoded_void = np.array([
            dic_void[key] for key in [
                ''.join([str(num) for num in vector]) for vector in [
                    i[0] for i in self.binary_array_sculpture.reshape(
                        self.void_dim * self.void_dim * self.void_dim, 1, 3).tolist()
                ]
            ]
        ]).reshape((self.void_dim, self.void_dim, self.void_dim))

        return decoded_void, decoded_color
