from perceptron import simple_perceptron
import argparse

if __name__ == '__main__':

    # Hyperparameters defaults
    EPOCHS = 100
    LEARNING_RATE = 0.1
    ACTIVATION = 'tanh'
    SPLIT = 0.2
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


    # Create instance of the simple_perceptron class
    p = simple_perceptron(epochs=epochs,
                          learning_rate=learning_rate,
                          activation=activation)

    # Get the data from a CSV file
    X,Y = p.read_input_data('train_data.csv')

    # Split into the train and validation sets
    Xtrain, Ytrain, Xvalid, Yvalid = \
        p.train_validation_split(X, Y, split=split,
                                 shuffle=shuffle)

    # Train of the perceptron
    p.train(Xtrain, Ytrain, Xvalid, Yvalid)

    # Save model to a file (with .model extension)
    p.save_model(input_filename[:-3] + 'model')
