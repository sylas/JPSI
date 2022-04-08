# Disable TensorFlow verbosity
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
import argparse

if __name__ == '__main__':

    # Hyperparameters defaults
    EPOCHS = 100
    SPLIT = 0.2
    LEARNING_RATE = 0.1
    ACTIVATION = 'tanh'
    SHUFFLE = False

    # Parse the arguments as parameters which can override defaults.
    # They can be also read from a file (@file.par)
    ap = argparse.ArgumentParser(fromfile_prefix_chars='@')
    ap.add_argument('-d', '--dataset', help='Path to train dataset',
        metavar='filename', required=True)
    ap.add_argument('-e', '--epochs', help='Number of epochs',
        type=int, default=EPOCHS, metavar='int')
    ap.add_argument('-l', '--learning_rate', help='Learning rate',
        type=float, default=LEARNING_RATE, metavar='float')
    ap.add_argument('-a', '--activation', help='Activation function',
        choices=['tanh', 'sigmoid', 'relu'], metavar='function', 
        default=ACTIVATION)
    ap.add_argument('-s', '--split', help='Train/validation split',
        type=float, default=SPLIT, metavar='float')
    ap.add_argument('-f', '--shuffle', help='Enable data shuffle',
        action='store_true', default=SHUFFLE)

    args = vars(ap.parse_args())

    input_filename = args['dataset']
    epochs = args['epochs']
    activation = args['activation']
    learning_rate = args['learning_rate']
    split = args['split']
    shuffle = args['shuffle']

    # Load train data from CSV file
    data = np.loadtxt(input_filename, delimiter=",")

    # Data normalization
    max_val = np.max(data)
    min_val = np.min(data)
    data = (data - min_val) / (max_val - min_val)

    # Divide into X and Y vectors
    Nin = data.shape[1] - 1
    X = data[:, :Nin]
    Y = data[:, Nin:]

    # Create model
    # https://keras.io/layers/core/
    # https://keras.io/activations/
    # One-directional model
    model = Sequential()
    # Simple perceptron: only 1 neuron with Nin inputs and no bias
    model.add(Dense(1, input_dim=Nin, activation=activation, use_bias=False))

    # Compile model
    # Loss functions: https://keras.io/losses/
    # Optimizers: https://keras.io/optimizers/
    model.compile(loss='mean_squared_error', 
            optimizer=SGD(learning_rate=learning_rate))

    # Fit the model
    H = model.fit(X, Y, epochs=epochs, validation_split=split, 
                  shuffle=shuffle, batch_size=1)

    # Save model to file
    model.save("model.h5")

    # Plot the train and validation losses
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(range(EPOCHS), H.history["loss"], label="Train loss")
    plt.plot(range(EPOCHS), H.history["val_loss"], label="Validation loss")
    plt.title("Train / validation loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.savefig("model.png")
