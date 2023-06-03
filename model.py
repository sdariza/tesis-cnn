import tensorflow as tf


class Autoencoder(tf.keras.Model):
    def __init__(self, name=None):
        # Call super().__init__() to initialize the base class
        super(Autoencoder, self).__init__(name=name)
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(filters=16, kernel_size=(
                    3, 3), activation='relu', padding='same', strides=2, input_shape=(32, 64, 1), kernel_initializer='normal'),
                tf.keras.layers.Conv2D(filters=8, kernel_size=(
                    3, 3), activation='relu', padding='same', strides=2, kernel_initializer='normal')
            ]
        )
        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2DTranspose(
                    filters=8, kernel_size=3, strides=2, activation='relu', padding='same'),
                tf.keras.layers.Conv2DTranspose(
                    filters=16, kernel_size=3, strides=2, activation='relu', padding='same'),
                tf.keras.layers.Conv2DTranspose(
                    filters=1, kernel_size=3, strides=1, activation='relu', padding='same'),
            ]
        )

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
