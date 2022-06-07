load_data = True

n_samples = 5000

void_dim = 56

BUFFER_SIZE = 5000

BATCH_SIZE = 32

EPOCHS = 500

noise_dim = 1024

num_examples_to_generate = 1

seed = tf.random.normal([num_examples_to_generate, noise_dim])
