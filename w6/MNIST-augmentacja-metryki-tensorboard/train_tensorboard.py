import os
# Disable TF messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# No of epochs for training
EPOCHS = 50

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

# Partition the data into training and validating splits
(Xtrain, Xvalid, Ytrain, Yvalid) = train_test_split(
                                   Xtrain, Ytrain, test_size=SPLIT)

Xtrain = Xtrain.reshape(Xtrain.shape[0], 28,28,1)
Xvalid = Xvalid.reshape(Xvalid.shape[0], 28,28,1)
Xtest = Xtest.reshape(Xtest.shape[0], 28,28,1) 

# Construct the image generator for data augmentation
aug = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=30, width_shift_range=0.1,
    height_shift_range=0.1, zoom_range=0.2,
    horizontal_flip=True, fill_mode="nearest")


# Configure TensorBoard
import tensorboard
logdir = 'logs'
# Start server
tb = tensorboard.program.TensorBoard()
tb.configure(argv=[None, '--logdir', logdir, '--bind_all'])
url = tb.launch()
print("TensorBoard address: " + url)

# Configure TensorBoard callback
tbCallback = tf.keras.callbacks.TensorBoard(
    log_dir=logdir, histogram_freq=1)


# Create model: one-directional network
model = tf.keras.models.Sequential()
# Flatten the input data
model.add(tf.keras.layers.Flatten(input_shape=(IMAGE_X, IMAGE_Y, IMAGE_DEPTH)))
# Hidden layer with relu activation
model.add(tf.keras.layers.Dense(500, activation='relu'))
# Output layer. Softmax activation gives probability for each class
model.add(tf.keras.layers.Dense(Nout, activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy', 
                       tf.keras.metrics.Precision(),
                       tf.keras.metrics.Recall()])

# Fit the model
H = model.fit(aug.flow(Xtrain, Ytrain), 
              validation_data=(Xvalid, Yvalid), 
              epochs=EPOCHS,
              callbacks=[tbCallback])

# Evaluate the model
print("Test score (loss and accuracy):")
test_score = model.evaluate(Xtest, Ytest)

loss = test_score[0]
accuracy = test_score[1]
precision = test_score[2]
recall = test_score[3]
f1 = 2*((precision*recall)/(precision+recall))

print("loss={:.4f}, accuracy={:.4f}, precision={:.4f}, recall={:.4f}, f1={:.4f}".format(
    loss, accuracy, precision, recall, f1)) 


# Save model to file
model.save("model.h5")

# Plot the train and validation losses
plt.style.use("ggplot")
plt.figure()
plt.plot(range(EPOCHS), H.history["loss"], label="Train loss")
plt.plot(range(EPOCHS), H.history["val_loss"], label="Validation loss")
plt.plot(range(EPOCHS), H.history["accuracy"], label="Train accuracy")
plt.plot(range(EPOCHS), H.history["val_accuracy"], label="Validation accuracy")
plt.plot(range(EPOCHS), H.history["precision"], label="Train precision")
plt.plot(range(EPOCHS), H.history["val_precision"], label="Validation precision")
plt.plot(range(EPOCHS), H.history["recall"], label="Train recall")
plt.plot(range(EPOCHS), H.history["val_recall"], label="Validation recall")
plt.title("Train / validation loss and accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss / Accuracy")
plt.legend(loc="center left")
plt.savefig("model.png")

