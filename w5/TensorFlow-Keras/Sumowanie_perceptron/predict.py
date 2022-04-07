# Disable TensorFlow verbosity
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from tensorflow.keras.models import load_model
import argparse


if __name__ == '__main__':

    # Max value for normalization
    MAX_VALUE = 10.0

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
    data /= MAX_VALUE

    # Divide into X and Y vectors
    Nin = data.shape[1] - 1
    X = data[:, :Nin]
    Y = data[:, Nin:]

    # Load model and print it's summary
    model = load_model(model_filename)
    model.summary()

    # Calculate predictions
    Ypredicted = model.predict(X)

    print("Test results:")
    # Print results (multiplied by MAX_VALUE as "unnormalization")
    for i in range(len(Ypredicted)):
        print('{} + {} = {:.3f} (expected {})'
                .format(X[i,0]*MAX_VALUE,
                        X[i,1]*MAX_VALUE,
                        Ypredicted[i,0]*MAX_VALUE,
                        Y[i,0]*MAX_VALUE))

