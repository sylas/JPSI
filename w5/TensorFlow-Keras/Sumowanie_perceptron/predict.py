# Disable TensorFlow verbosity
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow_addons.metrics.r_square import RSquare
import argparse


if __name__ == '__main__':

    # Max and min values for normalization
    MAX_VAL = 10.0
    MIN_VAL = 0

    # Parse the arguments. 
    # They can be also read from a file (@file.par)
    ap = argparse.ArgumentParser(fromfile_prefix_chars='@')
    ap.add_argument('-t', '--testset', help='Path to test dataset',
        metavar='filename', required=True)
    ap.add_argument('-m', '--model', help='Path to trained model',
        metavar='filename', required=True)

    args = vars(ap.parse_args())

    test_dataset_filename = args['testset']
    model_filename = args['model']

    # Load test data from CSV file
    data = np.loadtxt(test_dataset_filename, delimiter=",")

    # Normalize data
    data = (data - MIN_VAL) / (MAX_VAL - MIN_VAL)

    # Divide into X and Y vectors
    Nin = data.shape[1] - 1
    X = data[:, :Nin]
    Y = data[:, Nin:]

    # Load model and print it's summary
    model = load_model(model_filename)
    model.summary()

    # Calculate predictions
    Ypredicted = model.predict(X)

    # Metrics
    metric_rmse = RootMeanSquaredError()
    metric_r_squared = RSquare()
    metric_rmse.update_state(Y[:,0], Ypredicted[:,0])
    metric_r_squared.update_state(Y[:,0], Ypredicted[:,0])

    # "Unnormalization"
    Ypredicted = MIN_VAL + Y * (MAX_VAL - MIN_VAL)

    print("Test results:")
    for i in range(len(Ypredicted)):
        print('{:.3f} + {:.3f} = {:.3f} (expected {:.3f})'
                .format(X[i,0],
                        X[i,1],
                        Ypredicted[i,0],
                        Y[i,0]))

    print("RMSE = {:.3f}, R^2 = {:.3f}".format(metric_rmse.result().numpy(), 
                                               metric_r_squared.result().numpy()))
    