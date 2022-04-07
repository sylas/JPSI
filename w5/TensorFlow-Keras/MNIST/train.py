import os
# Disable TF messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# No of epochs for training
EPOCHS = 10

# Validation / train split
SPLIT = 0.2

# Input image dimensions
IMAGE_X = 28
IMAGE_Y = 28
IMAGE_DEPTH = 1

# Size of output vector (number of neurons in the output layer)
Nout = 10

# Load MNIST dataset as train and test sets
(Xtrain, Ytrain), (Xtest, Ytest) = tf.keras.datasets.mnist.load_data()

# Convert from uint8 to float32
Xtrain = Xtrain.astype('float32')
Xtest = Xtest.astype('float32')

# Normalize values to [0, 1]
Xtrain /= 255.0
Xtest /= 255.0

# Transform labels to 0-1 encoding
# 2 -> [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
# 6 -> [0, 0, 0, 0, 0, 0, 1, 0, 0, 0] etc.
Ytrain = tf.keras.utils.to_categorical(Ytrain, Nout)
Ytest = tf.keras.utils.to_categorical(Ytest, Nout)

# Create model: one-directional network
model = tf.keras.models.Sequential()
# Flatten the input data
model.add(tf.keras.layers.Flatten(input_shape=(IMAGE_X, IMAGE_Y, IMAGE_DEPTH)))
# Hidden layer with relu activation
model.add(tf.keras.layers.Dense(500, activation='relu'))
# Output layer. Softmax activation gives probability for each class
model.add(tf.keras.layers.Dense(Nout, activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
H = model.fit(Xtrain, Ytrain, epochs=EPOCHS, validation_split=SPLIT)

# Evaluate the model
print("Test score (loss and accuracy):")
test_score = model.evaluate(Xtest, Ytest)

# Save model to file
model.save("model.h5")

# Plot the train and validation losses
plt.style.use("ggplot")
plt.figure()
plt.plot(range(EPOCHS), H.history["loss"], label="Train loss")
plt.plot(range(EPOCHS), H.history["val_loss"], label="Validation loss")
plt.plot(range(EPOCHS), H.history["accuracy"], label="Train accuracy")
plt.plot(range(EPOCHS), H.history["val_accuracy"], label="Validation accuracy")
plt.title("Train / validation loss and accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss / Accuracy")
plt.legend(loc="center left")
plt.savefig("model.png")
