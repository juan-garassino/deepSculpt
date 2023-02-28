from tensorflow.keras.optimizers import Adam

from tensorflow.keras.optimizers.experimental import SGD

generator_optimizer = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999)

discriminator_optimizer = Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999)


import tensorflow as tf

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001,
                                     beta_1=0.9,
                                     beta_2=0.999,
                                     epsilon=1e-07)

optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.00005,
                                        rho=0.9)

optimizer = tf.keras.optimizers.Nadam(learning_rate=0.0002,
                                      beta_1=0.5,
                                      beta_2=0.999)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.001,
                                    momentum=0.9)

optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.01)
