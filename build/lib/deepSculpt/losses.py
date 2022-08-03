from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow import ones_like, zeros_like

cross_entropy = BinaryCrossentropy(from_logits=True)  # we take a BinaryCrossentropy

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(ones_like(real_output), real_output)
    # compares the predictions of the discriminator over real images to a matrix of [1s] | must have a tendency/likelihood to 1
    fake_loss = cross_entropy(zeros_like(fake_output), fake_output)
    # compares the predictions of the discriminator over generated images to a matrix of [0s] | must have a tendency/likelihood to 0
    total_loss = real_loss + fake_loss
    return total_loss  # Total loss

def generator_loss(fake_output):
    binary_cross_entropy = cross_entropy(ones_like(fake_output), fake_output)
    # the generator's output need to have a tendency to 1, We compare the discriminators decisions on the generated images to an array of [1s]
    return binary_cross_entropy
