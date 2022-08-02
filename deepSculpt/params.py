from tensorflow.random import normal

load_data = True

n_samples = 5000

void_dim = 24

BUFFER_SIZE = 5000

BATCH_SIZE = 32

EPOCHS = 5

noise_dim = 1024

num_examples_to_generate = 1

seed = normal([num_examples_to_generate, noise_dim])

BUCKET_NAME = ""

BUCKET_TRAIN_DATA_PATH = ""

MODEL_BASE_PATH = ""
