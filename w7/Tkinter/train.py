# Imports
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import preprocessing
import tensorflow_addons as tfa


def train_network(EPOCHS):
    #Initializations

    # Input data
    # CSV file name with train data
    DATA_FILENAME = "data.csv"
    # Filename for output (trained) model (HDF5 format)
    MODEL_FILENAME = "model.h5"
    # Validation / train split
    SPLIT = 0.2
    # Size of input vector (number of neurons in the input layer)
    Nin = 3
    # Size of output vector (number of neurons in the output layer)
    Nout = 1
    # Seed for random number generator - for reproductible results
    RANDOM_SEED = 100

    # Set random seed (TF >= 2.7, CPU only)
    tf.random.set_seed(100)

    # Load data
    data = np.loadtxt(DATA_FILENAME, delimiter=",")

    # Divide into X and Y vectors
    X = data[:, :Nin]
    Y = data[:, Nin:]

    # Normalize data
    X = min_max_scaler_X = preprocessing.MinMaxScaler().fit_transform(X)
    Y = min_max_scaler_Y = preprocessing.MinMaxScaler().fit_transform(Y)

    # Create model - one-directional network
    model = tf.keras.models.Sequential()
    # Input layer of dimension Nin
    model.add(tf.keras.layers.InputLayer(input_shape=(Nin, )))
    # Hidden layer: 2 neurons
    model.add(tf.keras.layers.Dense(2, activation='tanh'))
    # Hidden layer: 4 neurons
    model.add(tf.keras.layers.Dense(4, activation='tanh'))
    # Output layer: Nout neurons
    model.add(tf.keras.layers.Dense(Nout, activation='sigmoid'))

    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Fit the model
    #print(f"Train with EPOCHS = {EPOCHS}:") 
    H = model.fit(X, Y, epochs=EPOCHS, validation_split=SPLIT, verbose=0)

    # Evaluate the model
    #print("\nEvaluate model:")
    model.evaluate(X, Y, verbose=0)

    # Regression scores (R squared and RMSE)
    # We need predictions to calculate metrics
    Ypred = model.predict(X)

    metric = tfa.metrics.RSquare()
    metric.update_state(Y[:,0], Ypred[:,0])
    r2 = metric.result()

    metric = tf.keras.metrics.RootMeanSquaredError()
    metric.update_state(Y[:,0], Ypred[:,0])
    rmse = metric.result()

    # Save model to file
    model.save(MODEL_FILENAME)

    # Plot the train and validation losses
    plt.style.use("ggplot")
    plt.plot(range(EPOCHS), H.history["loss"], label="Train loss")
    plt.plot(range(EPOCHS), H.history["val_loss"], label="Validation loss")
    plt.title("Train / validation loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("model.png")
            
    return rmse, r2

if __name__ == "__main__":
    rmse, r2 = train_network(EPOCHS=100)
    print("\nScores:")
    print(f"RMSE = {rmse:.3f}")
    print(f"R^2  = {r2:.3f}")
    