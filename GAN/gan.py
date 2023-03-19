# Import the necessary libraries
import numpy as np
import tensorflow as tf
from tf.keras.layers import Dense, Reshape, Flatten
from tf.keras.models import Sequential
from tf.keras.optimizers import Adam
from tf.keras.datasets import mnist
import matplotlib.pyplot as plt
from tqdm import tqdm

# Load the Modbus dataset
data = np.load('D:\Major_Project\MP\IoT_Modbus.csv')

# Normalize the data
data = (data - np.min(data)) / (np.max(data) - np.min(data))

# Define the GAN model
generator = Sequential([
    Dense(128, input_shape=(100,)),
    tf.keras.layers.LeakyReLU(alpha=0.2),
    Dense(256),
    tf.keras.layers.LeakyReLU(alpha=0.2),
    Dense(512),
    tf.keras.layers.LeakyReLU(alpha=0.2),
    Dense(1024),
    tf.keras.layers.LeakyReLU(alpha=0.2),
    Dense(2048),
    tf.keras.layers.LeakyReLU(alpha=0.2),
    Dense(data.shape[1], activation='tanh')
])

discriminator = Sequential([
    Dense(2048, input_shape=(data.shape[1],)),
    tf.keras.layers.LeakyReLU(alpha=0.2),
    Dropout(0.3),
    Dense(1024),
    tf.keras.layers.LeakyReLU(alpha=0.2),
    Dropout(0.3),
    Dense(512),
    tf.keras.layers.LeakyReLU(alpha=0.2),
    Dropout(0.3),
    Dense(256),
    tf.keras.layers.LeakyReLU(alpha=0.2),
    Dropout(0.3),
    Dense(128),
    tf.keras.layers.LeakyReLU(alpha=0.2),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# Define the loss functions
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Define the optimizers
generator_optimizer = Adam(lr=0.0002, beta_1=0.5)
discriminator_optimizer = Adam(lr=0.0002, beta_1=0.5)

# Define the training loop
@tf.function
def train_step(generator, discriminator, real_data, batch_size, latent_dim):
    # Generate random noise
    noise = tf.random.normal([batch_size, latent_dim])

    # Generate fake data
    with tf.GradientTape() as gen_tape:
        generated_data = generator(noise, training=True)

        # Calculate the generator loss
        gen_loss = generator_loss(discriminator(generated_data, training=True))

    # Calculate the gradients of the generator loss
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)

    # Apply the gradients to the generator optimizer
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    # Train the discriminator
    with tf.GradientTape() as disc_tape:
        real_output = discriminator(real_data, training=True)
        fake_output = discriminator(generated_data, training=True)

        # Calculate the discriminator loss
        disc_loss = discriminator_loss(real_output, fake_output)

    # Calculate the gradients of the discriminator loss
    gradients_of_discriminator = disc_tape.gradient
