import tensorflow as tf


class Autoencoder(tf.keras.Model):
    def __init__(self, name=None):
        # Call super().__init__() to initialize the base class
        super(Autoencoder, self).__init__(name=name)
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(filters=512, kernel_size=(
                    3, 3), activation='relu', padding='same', input_shape=(32, 64, 1),),
                tf.keras.layers.Conv2D(filters=256, kernel_size=(
                    3, 3), activation='relu', padding='same',),
                tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),
                tf.keras.layers.Conv2D(filters=128, kernel_size=(
                    3, 3), activation='relu', padding='same',),
                tf.keras.layers.Conv2D(filters=64, kernel_size=(
                    3, 3), activation='relu', padding='same',),
                tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),

            ]
        )
        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, activation='relu', padding='same',),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, activation='relu', padding='same',),
                tf.keras.layers.UpSampling2D(size=(2, 2)),
                tf.keras.layers.Conv2D(
                    filters=128, kernel_size=3, activation='relu', padding='same',),
                tf.keras.layers.Conv2D(
                    filters=256, kernel_size=3, activation='relu', padding='same',),
                tf.keras.layers.UpSampling2D(size=(2, 2)),
                tf.keras.layers.Conv2D(
                    filters=512, kernel_size=3, activation='relu', padding='same',),
                tf.keras.layers.Conv2D(
                    filters=1, kernel_size=3, activation='relu', padding='same',),
            ]
        )

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
