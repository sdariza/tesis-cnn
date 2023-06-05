import tensorflow as tf
from model import Autoencoder
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def get_model():
    return Autoencoder(name='autoencoder')

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

model = get_model()
input_shape = (None,32,64,1)
model.build(input_shape= input_shape)
# model.compile(optimizer='adam', loss=tf.keras.losses.Huber(delta=10))
model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())
print(input_shape)



print(model.encoder.summary())
print(model.decoder.summary())

X = np.load('train_data/air.2018-2023_H0.npy')
Y = np.load('train_data/air.2018-2023_H6.npy')

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3, random_state=123)

history = model.fit(X_train,Y_train, epochs=150, shuffle=True, validation_data=(X_test,Y_test), batch_size=32, verbose=1)

# Save model
tf.keras.saving.save_model(model, 'gfgModel')
print('Model Saved!')

# Plot loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_function.png')