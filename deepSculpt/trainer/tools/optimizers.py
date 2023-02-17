from tensorflow.keras.optimizers import Adam

generator_optimizer = Adam(1e-5)  # SGD INSTEAD???   (Radford et al., 2015)

discriminator_optimizer = Adam(1e-5)  # SGD INSTEAD???  (Radford et al., 2015)
