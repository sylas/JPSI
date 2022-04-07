import numpy as np
import csv
import yaml
import matplotlib.pyplot as plt
import sys
from timeit import default_timer as timer


class simple_perceptron:
    '''
     Simple perceptron

            single output
                  ^
                  |
                  O
                / | \   weights
              Nin inputs
    '''

    def __init__(self, epochs=100, learning_rate=0.1, activation='tanh'):
        '''
        Constructor.
        Parameters:
            epochs - number of epochs (int)
            learning_rate - learning_rate (float)
            activation - activation function,
                    (str) of ['tanh', 'sigmoid', 'relu']
        '''

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.activation = activation
        random_seed = 100
        np.random.seed(random_seed)

        # Alternative way for constructor and setting the properties:
        #def __init__(self, **kwargs):
        #    # Reads input dictionary and uses it as a class properties:
        #    self.__dict__.update(kwargs)
        #    random_seed = 100
        #    np.random.seed(random_seed)


    def f(self, x):
        '''
        Calculates activation function at x.
        Parameters:
            x - an argument (float)
        Returns:
            value of the activation function at x (float)
        '''

        if self.activation == 'tanh':
            f = np.tanh(x)
        elif self.activation == 'sigmoid':
            f = 1 / (1 + np.exp(-x))
        elif self.activation == 'relu':
            f = x if x > 0 else 0
        else:
            sys.exit('Error: Unknown activation function.')

        return f


    def fp(self, x):
        '''
        Calculates derivative of the activation function at x.
        Parameters:
            x - an argument (float)
        Returns:
            value of the derivative to the activation function
            at x (float)
        '''

        if self.activation == 'tanh':
            fp = 1 - np.tanh(x)*np.tanh(x)
        elif self.activation == 'sigmoid':
            fp = np.exp(-x) / (1 + np.exp(-x))**2
        elif self.activation == 'relu':
            fp = 1 if x > 0 else 0
        else:
            sys.exit('Error: Unknown activation function.')

        return fp


    def read_input_data(self, filename, normalize=True):
        '''
        Reads input data (train or test) from the CSV file.
        Parameters:
            filename - CSV file name (string)
                CSV file format:
                    input1, input2, ..., output
                                    ...
                                    ...
            normalize - flag for data normalization (bool, optional)
        Sets:
            self.Nin = number of inputs of the perceptron (int)
        Returns:
            X - input training data (list)
            Y - output (expected) training data (list)
        '''

        # Read CSV data
        try:
            file = open(filename, 'rt')
        except FileNotFoundError:
            sys.exit('Error: data file does not exists.')

        dataset = csv.reader(file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)

        # Construct the X and Y lists. This is a simple perceptron with only one output,
        # so X should contain all data from all columns except the last one,
        # and Y - data from the last column only.
        X = []
        Y = []
        try:
            for line in dataset:
                X.append(line[0:-1])
                Y.append(line[-1])
        except ValueError:
            sys.exit('Error: Wrong format of the CSV file.')

        file.close()

        # Store the size of the input vector (Nin) as a class property
        self.Nin = len(X[0])

        if self.Nin == 0:
            sys.exit('Error: zero-length training vector.')

        # Normalize data, if requested
        if normalize:
            X,Y = self.normalize(X, Y)

        return X,Y


    def initialize_weights(self):
        '''
        Initialize weights with a random numbers from range [0,1).
        Parameters:
            Nin - number of inputs of the perceptron (int)
        Sets:
            self.weights property (list)
        Returns:
            None
        '''

        self.weights = np.random.random(self.Nin)

        # Alternative way - without using numpy:
        # self.weights = [random.random() for i in range(self.Nin)]
        #
        # or (in an expanded form):
        # self.weights = []
        # for i in range(self.Nin):
        #    self.weights.append(random.random())


    def train_validation_split(self, X, Y, split=0.2, shuffle=False):
        '''
        Splits the input vectors into the train and validation ones.
        Parameters:
            X - X-vector to be splitted (list)
            Y - Y-vector to be splitted (list)
            split - splitting factor (float in range [0.1-0.9], optional)
            shuffle - data shuffling flag (bool, optional)
        Returns:
            splitted Xtrain, Ytrain, Xvalid, Yvalid (lists)
        '''

        # Check if split is in a proper range, and adjust it if needed
        if split > 0.9:
            print('Warning: Wrong split, adjusting to 0.9.')
            split = 0.9

        if split < 0.1:
            print('Warning: Wrong split, adjusting to 0.1.')
            split = 0.1

        data_size = len(X)
        valid_data_size = int(split*data_size)

        if valid_data_size == 0:
            print("Validation set size is equal to zero, no validation will be performed!")

        # Generate list of random indexes indicating validation set
        valid_random_indexes = sorted(list(np.random.choice(
                                    data_size,
                                    size=valid_data_size,
                                    replace=False)))

        # Randomize data if requested
        if shuffle:
            randomize = np.arange(len(X))
            np.random.shuffle(randomize)
            X = X[randomize]
            Y = Y[randomize]

        Xvalid, Yvalid, Xtrain, Ytrain = [], [], [], []

        # Iterate over data and append them to the train or validation set
        for i in range(data_size):
            if i in valid_random_indexes:
                Xvalid.append(X[i])
                Yvalid.append(Y[i])
            else:
                Xtrain.append(X[i])
                Ytrain.append(Y[i])

        return Xtrain, Ytrain, Xvalid, Yvalid


    def normalize(self, X, Y):
        '''
        Normalizes the data and stores normalization parameters as properties.
        Parameters:
            X - X-vector to normalize (list)
            Y - Y-vector to normalize (list)
        Sets:
            self.min_val - minimum value used in normalization
            self.max_val - maximum value used in normalization
        Returns:
            normalized vectors X, Y (lists)
        '''

        # Find minimum and maximum values of in both vectors.
        # In subsequent runs, they will be used.
        if not hasattr(self, 'min_val'):
            self.min_val = min(np.amin(X), np.amin(Y))

        if not hasattr(self, 'max_val'):
            self.max_val = max(np.amax(X), np.amax(Y))

        # Normalization formulas (vector operations)
        X = (X - self.min_val) / (self.max_val - self.min_val)
        Y = (Y - self.min_val) / (self.max_val - self.min_val)

        return X, Y


    def unnormalize(self, *X):
        '''
        "Unnormalizes" vector(s), using previously determined minimum and maximum values.
        Parameters:
            X - tuple of vector(s) to normalize (lists)
        Returns:
            tuple of vectors of "unnormalized" vector(s) (lists)
        '''

        if hasattr(self, 'min_val') and hasattr(self, 'max_val'):
            Xout = []
            for Xsingle in X:
                # "Unnormalization" formula
                Xout.append([self.min_val + i * (self.max_val - self.min_val)
                    for i in Xsingle])
        else:
            # Cannot perform unnormalization, return original data
            print('Warning: Can not "unnormalize" data!')
            Xout = X

        return Xout


    def train(self, Xtrain, Ytrain, Xvalid, Yvalid):
        '''
        Trains the simple perceptron using the gradient method.
        Parameters:
            Xtrain - training (input) vector (list)
            Ytrain - training (output) vector (list)
            Xvalid - validating (input) vector (list)
            Yvalid - validating (output) vector (list)
        Returns:
            None
        '''

        start_time = timer()

        # Initialize weights
        self.initialize_weights()

        # Lists for storing RMSE for each epoch
        RMSE_train = []
        RMSE_valid = []

        # Iterate over epochs
        for epoch in range(self.epochs):
            print('Epoch = {}'.format(epoch+1))

            # Calculate output from the perceptron
            sumRMSE_train = 0
            for i in range(len(Xtrain)):
                # i - index of the input training pattern
                sumWeighted = 0
                for j in range(self.Nin):
                    # j - index of the weight, for the given input pattern
                    sumWeighted += self.weights[j]*Xtrain[i][j]
                Yout = self.f(sumWeighted)

                # Calculate weights change
                for j in range(self.Nin):
                    self.weights[j] += \
                        self.learning_rate * self.fp(sumWeighted) * (Ytrain[i]-Yout)*Xtrain[i][j]

                # Calculate contribution from the current epoch to the RMS on training set
                sumRMSE_train += (Yout-Ytrain[i])**2

            # Calculate and append RMS on training set
            RMSE_train.append(np.sqrt(sumRMSE_train / len(Xtrain)))
            print('RMSE (training set)   = {}'.format(RMSE_train[epoch]))

            if len(Xvalid) > 0:
                # Calculate RMS on validating set
                sumRMSE_valid = 0
                for i in range(len(Xvalid)):
                    sumWeighted = 0
                    for j in range(self.Nin):
                        sumWeighted += self.weights[j]*Xvalid[i][j]
                    Yout = self.f(sumWeighted)
                    sumRMSE_valid += (Yout-Yvalid[i])**2

                RMSE_valid.append(np.sqrt(sumRMSE_valid / len(Xvalid)))
                print('RMSE (validating set) = {}'.format(RMSE_valid[epoch]))

        print('\nTraining completed in {:.2f} seconds.'.format(timer()-start_time))

        # Save plot
        self.save_plot(RMSE_train, RMSE_valid)


    def test(self, Xtest):
        '''
        Test of the trained perceptron.
        Parameters:
            Xtest - test vector (list)
        Returns:
            Y - output from the perceptron (list)
        '''

        Y = []
        # Calculate output from the perceptron
        for i in range(len(Xtest)):
            # i - index of the input pattern
            sumWeighted = 0
            for j in range(self.Nin):
                # j - index of the weight, for the given input pattern
                sumWeighted += self.weights[j]*Xtest[i][j]
            Y.append(self.f(sumWeighted))

        return Y


    def save_plot(self, RMSE_train, RMSE_valid, filename='loss.png', show=True):
        '''
        Plots / saves / shows RMSE.
        Parameters:
            RMSE_train - RMSE on training set (list)
            RMSE_valid - RMSE on validating set (list)
            filename - file name for save plot (str, optional)
            show - display or not the plot (bool, optional)
        Returns:
            None
        '''

        plt.plot(RMSE_train, label='RMS (training set)')
        if len(RMSE_valid) > 0:
            plt.plot(RMSE_valid, label='RMS (validating set)')
        plt.legend()
        plt.title('Results of training of simple perceptron')
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.savefig(filename)
        if show:
            plt.show()
        print('RMSE plot has been saved to the file', filename)


    def save_model(self, filename):
        '''
        Saves the perceptron data into a YAML file.
        Parameters:
            filename - YAML file name (str)
        Returns:
            None
        '''

        data = {'nin':self.Nin,
                'epochs':self.epochs,
                'learning_rate':self.learning_rate,
                'activation':self.activation,
                'min_val':float(self.min_val),
                'max_val':float(self.max_val),
                'weights':self.weights.tolist()}

        with open(filename, 'w') as file:
            yaml.dump(data, file, default_flow_style=False)

        print('Model has been saved to file', filename)


    def load_model(self, filename):
        '''
        Loads the perceptron data from a YAML file.
        Parameters:
            filename - YAML file name (str)
        Returns:
            None
        '''

        try:
            with open(filename, 'r') as stream:
                data = yaml.load(stream, Loader=yaml.Loader)
        except FileNotFoundError:
            sys.exit('Error: Model file does not exists.')

        try:
            self.Nin = data['nin']
            self.epochs = data['epochs']
            self.learning_rate = data['learning_rate']
            self.activation = data['activation']
            self.min_val = np.float64(data['min_val'])
            self.max_val = np.float64(data['max_val'])
            self.weights = np.array(data['weights'])
        except KeyError:
            sys.exit('Error: Wrong format of the model file.')
