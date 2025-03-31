import numpy as np
from tensorflow import keras
from keras import layers

class GAN:
    """
    A simple Generative Adversarial Network (GAN) for generating synthetic data.

    Attributes:
        input_dim (int): The dimension of the input data.
        latent_dim (int): The dimension of the latent space for the generator.
        generator (keras.Model): The generator model.
        discriminator (keras.Model): The discriminator model.
        gan (keras.Model): The combined GAN model.
    """
    def __init__(self, input_dim:int = 10, latent_dim:int=100) -> None:
        """
        Initialize the GAN with generator and discriminator models.

        Args:
            input_dim (int): The dimension of the input data.
            latent_dim (int): The dimension of the latent space.
        """ 
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        self.gan = self._build_gan()
    
    def _build_generator(self) -> keras.Model:
        model = keras.Sequential([layers.Dense(128, activation="relu", input_dim=self.latent_dim),
                                  layers.Dense(self.input_dim, activation="tanh")])
        return model
    
    def _build_discriminator(self) -> keras.Model:
        self.discriminator.trainable = False
        model = keras.Sequential([self.generator, self.discriminator])
        model.compile(optimizer="adam", loss="binary_crossentropy")
        return model
    
    def train(self, real_data:np.ndarray, epochs:int=100, batch_size:int=32) -> None:
        """
        Train the GAN model.

        Args:
            real_data (np.ndarray): The real data to train on.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
        """
        for epoch in range(epochs):
            # Train discriminator
            idx = np.random.randint(0, real_data.shape[0], batch_size)
            real_samples = real_data[idx]
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            fake_samples = self.generator.predict(noise, verbose = 0)
            d_loss_real = self.discriminator.train_on_batch(real_samples, np.ones((batch_size, 1)))
            d_loss_fake = self.discriminator.train_on_batch(fake_samples, np.zeros((batch_size, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            # Train generator
            noise = np.random_normal(0,1,(batch_size, self.latent_dim))
            g_loss = self.gan.train_on_batch(noise, np.ones((batch_size, 1)))
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, D_Loss: {d_loss}, G_loss: {g_loss}")
    def generate(self, num_samples:int) -> np.ndarray:
        """
        Generate synthetic data using the trained generator.

        Args:
            num_samples (int): Number of synthetic samples to generate.

        Returns:
            np.ndarray: The generated synthetic data.
        """
        noise = np.random.noise(0, 1, (num_samples, self.latent_dim))
        synthetic_data = self.generator.predict(noise, verbose = 0)
        return synthetic_data
