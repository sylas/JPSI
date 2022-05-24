'''
Program trains dense / fully connected neural network for regression problems,
using Keras/Tensorflow platform and the k-fold cross-validation method.

Run '{} -h' for full list and description of the input parameters.
'''

print(__doc__.format(__file__))

### DEFAULTS ###
# Any of them can be overridden by passing command-line parameters

# No of epochs for training
EPOCHS = 1000

# Dense (fully connected) or dropout layers definition:
# [[no. of neurons, activation], [rate, 'dropout'], [...]]
# Activations: https://keras.io/api/layers/activations/
# No. of neurons in the last layer is used to determine size of the Y-vector
LAYERS = [[9, 'tanh'], [7, 'sigmoid'], [1, 'tanh']]

# Loss function; https://keras.io/api/losses/
LOSS = "mean_squared_error"

# Optimizer; https://keras.io/api/optimizers/
OPTIMIZER = "adam"

# Batch size - no. of samples taken for calculated averaged gradient.
# Lower is (generally) more accurate, but slower.
# -1 sets batch size as the number of samples.
BATCH_SIZE = 1

# Number of subsequent program runs for cross-validation
# (k in the k-fold method), each for different (random) validation set
FOLDS = 10

# Validation data size, relative to the whole set
# 0.2 is the rule of thumb, but 0.1 may be enough if FOLDS is quite high
VAL_SPLIT = 0.2

# During normalization, Y values are scaled into the [0, 1] range
# by usual mapping [y_min, y_max] -> [0, 1].
# By using extension factor, new mapping is intruduced:
# [y_min-Y_MIN_FACTOR*y_min, y_max+Y_MAX_FACTOR*y_max] -> [0, 1]
Y_MIN_FACTOR = 0.0
Y_MAX_FACTOR = 0.0

# Seed for random numbers generator
RANDOM_SEED = 100

# Directory for saving results
OUTPUT_DIR_BASE = "results"


### IMPORTS ###

import os
import sys
from timeit import default_timer as timer
import datetime
import numpy as np
import argparse
import json
import psutil
from collections import defaultdict

import matplotlib
matplotlib.use('Agg') # Enable plotting backend for writing to a file
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter1d
from math import sqrt


### Helper functions ###

def printf(what, end="\n"):
    '''Prints argument to 'stdout' and file 'outfile' (must be opened).'''
    print(what, end=end)
    try:
        print(what, end=end, file=outfile)
    except (ValueError, NameError):
        pass


def open_port(port_start, port_end):
    '''Probes for free port, and returns first available from a given range.'''
    for port in range(port_start, port_end):
        cnts = defaultdict(int)
        for c in psutil.net_connections():
            c_port = c.laddr[1]
            if c_port != port:
                continue
            status = c.status.lower()
            cnts[status] += 1
        if cnts["listen"] == 0:
            break
    return port


### PARSING ARGUMENTS ###

# Parse the arguments as parameters which can override defaults.
# They can be also read from a file (by using @file.par)
ap = argparse.ArgumentParser(fromfile_prefix_chars='@')
ap.add_argument("--dataset",
                help="Path to train dataset (in a CSV format: x1, x2, ..., y1, y2, ...)",
                metavar='filename', required=True)
ap.add_argument("--output_dir", help="Directory, where results will be stored",
                metavar='dirname', required=False, default=OUTPUT_DIR_BASE)
ap.add_argument("--epochs", type=int, help="Number of epochs", default=EPOCHS, metavar='int')
ap.add_argument("--optimizer", choices=['sgd', 'rmsprop', 'adam', 'nadam',
                'adamax', 'adagrad', 'adadelta', 'ftrl'],
                help="Optimizer", default=OPTIMIZER)
ap.add_argument("--loss", choices=['mean_squared_error', 'mean_absolute_error',
                'logcosh', 'huber_loss', 'cosine_similarity',
                'mean_squared_logarithmic_error', 'mean_absolute_percentage_error'],
                help="Loss function", default=LOSS)
ap.add_argument("--layers", nargs='+',
                help="No. of neurons and activation functions in subsequent layers, "
                "e.g. 4 tanh 3 relu 1 sigmoid", metavar='int string')
ap.add_argument("--folds", type=int, help="Number of folds for different train/validation splits",
                default=FOLDS, metavar='int')
ap.add_argument("--batch_size", type=int, help="Batch size (-1 if equal to training set size)",
                default=BATCH_SIZE, metavar='int')
ap.add_argument("--val_split", type=float, help="Validation split", default=VAL_SPLIT,
                metavar='float')
ap.add_argument("--y_min_factor", type=float, help="Extension factor for y_min",
                default=Y_MIN_FACTOR, metavar='float')
ap.add_argument("--y_max_factor", type=float, help="Extension factor for y_max",
                default=Y_MAX_FACTOR, metavar='float')
ap.add_argument("--random_seed", type=int, help="Random seed", default=RANDOM_SEED, metavar='int')
ap.add_argument("--gpu", action='store_true', help="Use GPU", default=False)
ap.add_argument("--keep_weights", action='store_true', help="Keep weights between folds",
                default=False)
ap.add_argument("--cb_tensorboard", action='store_true', help="Enable TensorBoard callback",
                default=False)
ap.add_argument("--cb_checkpoint", action='store_true', help="Enable ModelCheckpoint callback",
                default=False)
ap.add_argument("--cb_csv_logger", action='store_true', help="Enable CSVLogger callback",
                default=False)
ap.add_argument("--cb_early_stopping", action='store_true', help="Enable EarlyStopping callback",
                default=False)
ap.add_argument("--cb_reduce_lr", action='store_true', help="Enable ReduceLROnPlateau callback",
                default=False)
args = vars(ap.parse_args())

# Parse layers
if args["layers"] is not None:
    if len(args["layers"]) == 1:
        # Split into separate elements if needed (e.g. if read from .par file)
        args["layers"] = [words for segments in args["layers"] for words in segments.split()]
    layers=[]
    for arg in args["layers"]:
        try:
            layers.append(int(arg)) # No of neurons in layer, or...
        except ValueError:
            try:
                layers.append(float(arg)) # dropout rate, or...
            except ValueError:
                layers.append(arg) # activation function or 'dropout'

    # Reshape into [[,],[,]]
    layers = [layers[i:i+2] for i in range(0, len(layers), 2)]
    # Check if total number of arguments is even
    args_layers = sum(len(x) for x in layers)
    if args_layers%2 != 0:
        ap.error('Fatal error: Incorrect definition of NN layers.')
else:
    layers = LAYERS

input_filename = args["dataset"]
output_dir_base = args["output_dir"]
epochs = args["epochs"]
optimizer = args["optimizer"]
loss = args["loss"]
y_min_factor = args["y_min_factor"]
y_max_factor = args["y_max_factor"]
folds = args["folds"]
cb_tensorboard_enable = args["cb_tensorboard"]
cb_checkpoint_enable = args["cb_checkpoint"]
cb_csv_logger_enable = args["cb_csv_logger"]
cb_early_stopping_enable = args["cb_early_stopping"]
cb_reduce_lr_enable = args["cb_reduce_lr"]
random_seed = args["random_seed"]
use_gpu = args["gpu"]
keep_weights = args["keep_weights"]
val_split = args["val_split"]
batch_size = args["batch_size"]


### TENSORFLOW LOAD AND CONFIG ###

print("Initializations...\n")

# Disable GPU / CUDA visibility
if not use_gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Disable TF loading and runtime warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# For reproductible results
tf.keras.utils.set_random_seed(random_seed)

###  DATA LOADING AND PREPARATION ###

# Construct the name of the directory for results
current_date_time = datetime.datetime.now()
output_dir = output_dir_base + os.path.sep + str(current_date_time.strftime("%Y-%m-%d_%H-%M-%S"))

# Load train dataset
try:
    dataset = np.loadtxt(input_filename, delimiter=",")
except FileNotFoundError:
    print("Fatal error: File with dataset not found.")
    sys.exit(1)
    

# Y-vector size, determined from number of neurons in the last layer.
# Other from 1 not yet tested, might not work out of the box.
y_data_size = layers[-1][0]

# X-vector size,
# determined as the length of single data line, minus y_data_size
x_data_size = len(dataset[0]) - y_data_size

if x_data_size < 1:
    print("Fatal error: Incorrect length of X train vector.")
    sys.exit(1)
if y_data_size < 1:
    print("Fatal error: Incorrect length of Y train vector.")
    sys.exit(1)

# Column at which the Y variables start (counting from 0)
y_data_start = x_data_size

# Find minimum/maximum values in X and Y data, for further normalization
x_max = np.amax(dataset[:,0:x_data_size], axis=0)
y_max = np.amax(dataset[:,y_data_start:], axis=0)
x_min = np.amin(dataset[:,0:x_data_size], axis=0)
y_min = np.amin(dataset[:,y_data_start:], axis=0)

# Extend the Y range by a factor
for i in range(y_data_size):
    y_min[i] = y_min[i] - y_min_factor*y_min[i]
    y_max[i] = y_max[i] + y_max_factor*y_max[i]

# Create the output directory
try:
    os.makedirs(output_dir)
except OSError:
    print("Fatal error: Cannot create output directory (probably wrong permisions).")
    sys.exit(1)
output_filename_core = output_dir + os.path.sep + input_filename[:-4]

# Summary file
outfile = open(output_filename_core+"-out.txt","w")

# Generate random indexes of input data set used for validating
# (excluded from training)
valIndexes = []
no_all_samples = dataset.shape[0]
no_val_samples = int(val_split * no_all_samples)
no_train_samples = no_all_samples - no_val_samples
for i in range(folds):
    # Watch if the newly drawn random list is unique, if not - draw again
    is_randomlist_unique = False
    while not is_randomlist_unique:
        randomlist = sorted(list(np.random.choice(no_all_samples, size=no_val_samples,
                                                  replace=False)))
        if randomlist not in valIndexes:
            is_randomlist_unique = True
    valIndexes.append(randomlist)

# Batch size as the number of training data
if batch_size == -1:
    batch_size = no_train_samples

# Array for store weights between folds
stored_weights = []

# Calculate number of trainable parameters
trainable_params = 0
for i in range(len(layers)):
    trainable_params += layers[i][0]
    if i == 0:
        trainable_params += x_data_size*layers[i][0]
    else:
        trainable_params += layers[i][0]*layers[i-1][0]


### CALLBACKS CONFIG ###

class PercentPrinterCB(tf.keras.callbacks.Callback):
    ''' Prints losses 100 times per fold '''
    printCounter = 0
    printCounterMax = epochs/100
    lastLoss = 1.0
    lastValLoss = 1.0
    def on_train_begin(self, logs={}):
        PercentPrinterCB.start = timer()
    def on_train_end(self, logs={}):
        print("\r...finished in {:.2f} seconds.{:90}".format(timer()-PercentPrinterCB.start, " "))
    def on_epoch_end(self, epoch, logs={}):
        PercentPrinterCB.printCounter += 1
        loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        if PercentPrinterCB.printCounter == PercentPrinterCB.printCounterMax:
            print("\r[{:2.0f}%] loss = {:13.10f} ({:7.2f}%) | val_loss = {:13.10f} ({:7.2f}%) | "
                  "val_loss/loss ratio = {:6.2f}       "
            .format(epoch/epochs*100, loss,
                    (loss-PercentPrinterCB.lastLoss)/PercentPrinterCB.lastLoss*100, val_loss,
                    (val_loss-PercentPrinterCB.lastValLoss)/PercentPrinterCB.lastValLoss*100,
                     val_loss/loss), end="", flush=True)
            PercentPrinterCB.printCounter = 0
            PercentPrinterCB.lastLoss = loss
            PercentPrinterCB.lastValLoss = val_loss

class DelayedEarlyStopping(tf.keras.callbacks.EarlyStopping):
    ''' Custom EarlyStopping, activated only after reaching delay_epoch epoch '''
    def __init__(self, monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto',
                 restore_best_weights=False, delay_epoch=100):
        super(DelayedEarlyStopping, self).__init__(monitor=monitor, patience=patience, mode=mode,
                                    min_delta=min_delta, restore_best_weights=restore_best_weights)
        self.delay_epoch = delay_epoch
    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.delay_epoch:
            super().on_epoch_end(epoch, logs)

# List of callbacks to use
callbacks_list = [PercentPrinterCB()]

# TensorBoard callback; log_dir will be added in a fold loop
if cb_tensorboard_enable:
    logdir = output_dir + os.path.sep + 'logs'
    try:
        os.mkdir(logdir)
    except OSError:
        print("Warning: Can't create log dir for TensorBoard, disabling callback.")
        cb_tensorboard_enable = False
    else:
        # Start server
        from tensorboard import program
        tb_server = program.TensorBoard()        
        tb_port = str(open_port(6006,6100))
        tb_server.configure(argv=[None, '--logdir', logdir, '--bind_all', '--port='+tb_port])
        url = tb_server.launch()
        cb_tensorboard = tf.keras.callbacks.TensorBoard(histogram_freq=0, embeddings_freq=0,
                profile_batch=0, write_graph=False, write_images=False)
        callbacks_list.append(cb_tensorboard)
        print("Tensorboard callback enabled; address: ", url)

# ReduceLROnPlateau - reduce learning rate when a metric has stopped improving
if cb_reduce_lr_enable:
    cb_reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9,
                                          patience=2, min_lr=1e-5)
    callbacks_list.append(cb_reduce_lr)
    print("ReduceLROnPlateau callback enabled.")

# (Delayed) EarlyStopping - stop training when a monitored metric has stopped improving
if cb_early_stopping_enable:
    cb_early_stopping = DelayedEarlyStopping(monitor='val_loss', patience=100,
                                    restore_best_weights=True, delay_epoch=epochs/5)
    callbacks_list.append(cb_early_stopping)
    print("(Delayed) EarlyStopping callback enabled.")

# The ModelCheckpoint callback; filepath will be completed in a fold loop
if cb_checkpoint_enable:
    # Create empty object, parameters will be provided later, for each fold
    cb_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath="")
    callbacks_list.append(cb_checkpoint)
    print("ModelCheckpoint callback enabled.")

# The CSVLogger callback; filename will be completed in a fold loop
if cb_csv_logger_enable:
    cb_csv_logger = tf.keras.callbacks.CSVLogger(filename="")
    callbacks_list.append(cb_csv_logger)
    print("CSVLogger callback enabled.")


### MAIN PART OF CROSS_VALIDATION TRAINING ###

# Print header
print("Date time:     {}".format(current_date_time.strftime("%Y-%m-%d %H.%M.%S")), file=outfile)
printf("\nDataset:       {}".format(input_filename))
printf("Dataset size:  {} (training: {}, validating: {})"
    .format(no_all_samples, no_train_samples, no_val_samples))
printf("X vector size: {}".format(x_data_size))
printf("Y vector size: {}".format(y_data_size))
printf("Dense layers:  {}".format(layers))
printf("Weights:       {}".format(trainable_params))
printf("Epochs:        {}".format(epochs))
printf("Optimizer:     {}".format(optimizer))
printf("Loss function: {}".format(loss))
printf("Batch size:    {}".format(batch_size))
printf("Folds:         {}".format(folds))
printf("Keep weights:  {}".format(keep_weights))
printf("Random seed:   {}".format(random_seed))
printf("Min/max values:")
printf("       x_min = {}".format(list(x_min)))
printf("       x_max = {}".format(list(x_max)))
printf("       y_min = {}".format(list(y_min)))
printf("       y_max = {}".format(list(y_max)))
outfile.flush();

for fold in range(1, folds+1):
    validationIndexes = valIndexes[fold-1]

    # Set fold-dependent parameters of TensorBoard callback
    if cb_tensorboard_enable:
        cb_tensorboard.log_dir = logdir + \
            os.path.sep + "f" + "{:02d}".format(fold)

    # Set fold-dependent parameters of ModelCheckpoint callback
    if cb_checkpoint_enable:
        cb_checkpoint_filename = output_filename_core + "_f{:02d}-best.h5".format(fold)
        cb_checkpoint.__init__(filepath=cb_checkpoint_filename, verbose=0, monitor='val_loss',
                                                       save_best_only=True)

    # Set fold-dependent parameters of CSVLogger callback
    # Note: this callback is redundant to the TensorBoard
    if cb_csv_logger_enable:
        cb_csv_logger_filename = output_filename_core + "_f{:02d}.csv".format(fold)
        cb_csv_logger.__init__(filename=cb_csv_logger_filename)

    # Divide into the train and validation sets
    trainIndexes = [i for i in range(len(dataset)) if i not in validationIndexes]
    XYt = np.copy(dataset[trainIndexes, ])
    XYv = np.copy(dataset[validationIndexes, ])
    Xtrain = XYt[:, 0:x_data_size]
    Ytrain = XYt[:, y_data_start:]
    Xval   = XYv[:, 0:x_data_size]
    Yval   = XYv[:, y_data_start:]

    # Data normalization
    # Normalize Y (and keep the original data)
    do_y_rescaling = False
    Ytrain_orig = np.copy(Ytrain)
    Yval_orig = np.copy(Yval)
    for j in range(y_data_size):
        if y_max[j] > 1 or y_min[j] < 0:
            do_y_rescaling = True
            Ytrain[:,j] = (Ytrain[:,j] - y_min[j]) / (y_max[j] - y_min[j])
            Yval[:,j] = (Yval[:,j] - y_min[j]) / (y_max[j] - y_min[j])

    # Normalize X (and keep the original data)
    Xtrain_orig = np.copy(Xtrain)
    Xval_orig = np.copy(Xval)
    for j in range(x_data_size):
        if x_max[j] > 1 or x_min[j] < 0:
            Xtrain[:,j] = (Xtrain[:,j] - x_min[j]) / (x_max[j] - x_min[j])
            Xval[:,j] = (Xval[:,j] - x_min[j]) / (x_max[j] - x_min[j])

    # Find max number of digits for X (for more compact display only)
    max_len_x = [0 for i in range(x_data_size)]
    for i in range(len(Xtrain_orig[0])):
        for j in np.take(Xtrain_orig, indices=i, axis=1):
            if len(str(j)) > max_len_x[i]:
                max_len_x[i] = len(str(j))
    for i in range(len(Xval_orig[0])):
        for j in np.take(Xval_orig, indices=i, axis=1):
            if len(str(j)) > max_len_x[i]:
                max_len_x[i] = len(str(j))

    printf("\nTraining (fold {}/{})...".format(fold, folds))
    printf("Validating set indexes = {}".format(validationIndexes))
    printf("Training set indexes   = {}".format(trainIndexes))

    # Define and compile model
    model = tf.keras.models.Sequential()
    for i,layer in enumerate(layers):
        if i == 0: # First layer
            if layer[1] == "dropout":
                model.add(tf.keras.layers.Dropout(layer[0]), input_dim=x_data_size)
            else:
                model.add(tf.keras.layers.Dense(layer[0], input_dim=x_data_size,
                                                activation=layer[1]))
                # kernel_regularizer=tf.keras.regularizers.l2(factor)
        elif i == (len(layers)-1): # Last layer
            model.add(tf.keras.layers.Dense(layer[0], activation=layer[1]))                                
        else: # Second and higher hidden layers
            if layer[1] == "dropout":
                model.add(tf.keras.layers.Dropout(layer[0]))
            else:
                model.add(tf.keras.layers.Dense(layer[0], activation=layer[1]))
    model.compile(loss=loss, optimizer=optimizer)

    # Restore weights from previous fold
    if keep_weights:
        if stored_weights:
            for i,layer in enumerate(model.layers):
                layer.set_weights(stored_weights[i])

    # Train the model
    h = model.fit(Xtrain, Ytrain, validation_data=(Xval, Yval), batch_size=batch_size,
                  epochs=epochs, verbose=0, callbacks=callbacks_list)

    if cb_early_stopping_enable:
        printf("Early-stop final epoch = {}".format(len(h.history["loss"])))

    # Save model to file
    model.save(output_filename_core+"_f{:02d}.h5".format(fold))

    # Evaluate model and make predictions
    eval_valid = model.evaluate(Xval, Yval, verbose=0)
    printf("Loss on validating set = {:.15f}".format(eval_valid))
    eval_train = model.evaluate(Xtrain, Ytrain, verbose=0)
    printf("Loss on training set   = {:.15f}".format(eval_train))
    eval_complete = model.evaluate(np.concatenate((Xtrain, Xval)), np.concatenate((Ytrain, Yval)), verbose=0)
    printf("Loss on complete set   = {:.15f}".format(eval_complete))

    pred_valid = model.predict(Xval)
    pred_train = model.predict(Xtrain)

    if cb_checkpoint_enable:
        # Load best model, evaluate and predict
        model_best = tf.keras.models.load_model(cb_checkpoint_filename)
        eval_valid_best = model_best.evaluate(Xval, Yval, verbose=0)
        printf("Loss on validating set = {:.15f} (best epoch)".format(eval_valid_best))
        eval_train_best = model_best.evaluate(Xtrain, Ytrain, verbose=0)
        printf("Loss on training set   = {:.15f} (best epoch)".format(eval_train_best))
        eval_complete_best = model_best.evaluate(np.concatenate((Xtrain, Xval)), np.concatenate((Ytrain, Yval)), verbose=0)
        printf("Loss on complete set   = {:.15f} (best epoch)".format(eval_complete_best))

        pred_valid_best = model_best.predict(Xval)
        pred_train_best = model_best.predict(Xtrain)

    # Scale back predictions, if necessary
    if do_y_rescaling:
        pred_valid =  [i*(y_max-y_min)+y_min for i in pred_valid]
        pred_train =  [i*(y_max-y_min)+y_min for i in pred_train]
        if cb_checkpoint_enable:
            pred_valid_best = [i*(y_max-y_min)+y_min for i in pred_valid_best]
            pred_train_best = [i*(y_max-y_min)+y_min for i in pred_train_best]

    # Store weights for the next fold
    if keep_weights:
        stored_weights = []
        if cb_checkpoint_enable:
            for layer in model_best.layers:
                stored_weights.append(layer.get_weights())
        else:
            for layer in model.layers:
                stored_weights.append(layer.get_weights())

    # Print predictions header
    print("Predictions:")
    for j in range(x_data_size):
        printf("x{:<{}d}".format(j,max_len_x[j]-1), end="")
    if cb_checkpoint_enable:
        for j in range(y_data_size):
            printf("         y{}    y{}best     y{}exp       d{}   d{}best".format(j,j,j,j,j),
                  end="")
    else:
        for j in range(y_data_size):
            printf("         y{}     y{}exp       d{}".format(j,j,j), end="")
    printf("")

    # RMSE initializatons
    mse, mse_v, mse_t = 0, 0, 0
    if cb_checkpoint_enable:
        mse_best, mse_best_v, mse_best_t = 0, 0, 0

    # "Trap" for the case when Y is zero, so relative error can not be calculated
    print_relative_error = False if ((0 in Ytrain_orig) or (0 in Yval_orig)) else True
    ###
    print_relative_error = False
    err_unit = "%" if print_relative_error else ""

    # Print predictions on validation samples
    for i in range(no_val_samples):
        diff = []
        for j in range(y_data_size):
            if print_relative_error:
                diff.append(100*(pred_valid[i][j]-Yval_orig[i,j])/Yval_orig[i,j])
            else:
                diff.append(pred_valid[i][j]-Yval_orig[i,j])
            mse   += (pred_valid[i][j]-Yval_orig[i,j])**2
            mse_v += (pred_valid[i][j]-Yval_orig[i,j])**2
        if cb_checkpoint_enable:
            diff_best = []
            for j in range(y_data_size):
                if print_relative_error:
                    diff_best.append(100*(pred_valid_best[i][j]-Yval_orig[i,j])/Yval_orig[i,j])
                else:
                    diff_best.append(pred_valid_best[i][j]-Yval_orig[i,j])
                mse_best   += (pred_valid_best[i][j]-Yval_orig[i,j])**2
                mse_best_v += (pred_valid_best[i][j]-Yval_orig[i,j])**2
        for j in range(x_data_size):
            printf("{:<{}.1f} ".format(Xval_orig[i,j],max_len_x[j]), end="")
        if cb_checkpoint_enable:
            for j in range(y_data_size):
                printf("{:>10.2f}{:>10.2f}{:>10.2f}{:>8.2f}{}{:>8.2f}{}"
                    .format(pred_valid[i][j], pred_valid_best[i][j], Yval_orig[i,j], diff[j],
                            err_unit, diff_best[j], err_unit), end="")
        else:
            for j in range(y_data_size):
                printf("{:>10.2f}{:>10.2f}{:>8.2f}{}".format(pred_valid[i][j],Yval_orig[i,j],
                      diff[j], err_unit), end="")
        printf(" [val]")

    # Print predictions on train samples
    for i in range(no_train_samples):
        diff = []
        for j in range(y_data_size):
            if print_relative_error:
                diff.append(100*(pred_train[i][j]-Ytrain_orig[i,j])/Ytrain_orig[i,j])
            else:
                diff.append(pred_train[i][j]-Ytrain_orig[i,j])
            mse   += (pred_train[i][j]-Ytrain_orig[i,j])**2
            mse_t += (pred_train[i][j]-Ytrain_orig[i,j])**2
        if cb_checkpoint_enable:
            diff_best = []
            for j in range(y_data_size):
                if print_relative_error:
                    diff_best.append(100*(pred_train_best[i][j]-Ytrain_orig[i,j])/Ytrain_orig[i,j])
                else:
                    diff_best.append(pred_train_best[i][j]-Ytrain_orig[i,j])
                mse_best   += (pred_train_best[i][j]-Ytrain_orig[i,j])**2
                mse_best_t += (pred_train_best[i][j]-Ytrain_orig[i,j])**2
        for j in range(x_data_size):
            printf("{:<{}.1f} ".format(Xtrain_orig[i,j], max_len_x[j]), end="")
        if cb_checkpoint_enable:
            for j in range(y_data_size):
                printf("{:>10.2f}{:>10.2f}{:>10.2f}{:>8.2f}{}{:>8.2f}{}"
                    .format(pred_train[i][j], pred_train_best[i][j], Ytrain_orig[i,j], diff[j],
                            err_unit, diff_best[j], err_unit), end="")
        else:
            for j in range(y_data_size):
                printf("{:>10.2f}{:>10.2f}{:>8.2f}{}".format(pred_train[i][j], Ytrain_orig[i,j],
                      diff[j], err_unit), end="")
        printf("")

    # RMSE calculations
    rmse   = round(sqrt(mse/no_all_samples),3)
    rmse_v = round(sqrt(mse_v/no_val_samples),3)
    rmse_t = round(sqrt(mse_t/no_train_samples),3)
    printf("RMSE on validating/training/complete set = {} / {} / {}".format(rmse_v, rmse_t, rmse))
    if cb_checkpoint_enable:
        rmse_best   = round(sqrt(mse_best/no_all_samples),3)
        rmse_best_v = round(sqrt(mse_best_v/no_val_samples),3)
        rmse_best_t = round(sqrt(mse_best_t/no_train_samples),3)
        printf("RMSE on validating/training/complete set = {} / {} / {} (best epoch)".format(
                rmse_best_v, rmse_best_t, rmse_best))
    outfile.flush();

    # PLOTS
    Xrange = range(min(epochs, len(h.history["loss"])))
    plt.figure()

    # Calculate upper outliers to remove them from the graph
    Q1 = min(np.percentile(h.history["val_loss"], 25), np.percentile(h.history["loss"], 25))
    Q3 = max(np.percentile(h.history["val_loss"], 75), np.percentile(h.history["loss"], 75))
    IQR = Q3 - Q1
    max_range = Q3 + 1.5*IQR
    axes = plt.gca()
    axes.set_ylim([0, max_range])

    # Smoothing loss data
    Yrange_loss = gaussian_filter1d(h.history["loss"], sigma=5)
    plt.plot(Xrange, Yrange_loss, label="train")
    try:
        Yrange_val_loss = gaussian_filter1d(h.history["val_loss"], sigma=5)
        plt.plot(Xrange, Yrange_val_loss, label="validation")
    except KeyError:
        pass
    plt.title("Train / validation loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(output_filename_core+"_f{:02d}.png".format(fold))

# Print model summary to a file
outfile.write("\nModel summary:")
model.summary(print_fn=lambda x: outfile.write(x + '\n'))
model_json = json.loads(model.to_json())
outfile.write(json.dumps(model_json, indent=2))
outfile.close()

print("\nAll done.")
