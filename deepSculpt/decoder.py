class Decoder(Sculptor):

    def __init__(self, void, style):
        self.void = void

    def decoder(binary_array, classes, void_dim):

        dic_color = {'000':classes[0],'001':classes[1],'010':classes[2],'011':classes[3],'100':classes[4],'101':classes[5],"110":classes[6],"111":classes[7]}

        dic_void = {'000':1,'001':1,'010':1,'011':1,'100':1,'101':0,"110":0,"111":0}

        decoded_color = np.array([dic_color[key] for key in [''.join([str(num) for num in vector]) for vector in [i[0] for i in binary_array.reshape(void_dim*void_dim*void_dim,1,3).tolist()]]]).reshape((void_dim,void_dim,void_dim))

        decoded_void = np.array([dic_void[key] for key in [''.join([str(num) for num in vector]) for vector in [i[0] for i in binary_array.reshape(void_dim*void_dim*void_dim,1,3).tolist()]]]).reshape((void_dim,void_dim,void_dim))

        return decoded_void, decoded_color
