print("Hyperparameter search using Keras tuner.")
print("Using RandomSearch tuner by default; run with -H for Hyperband tuner.")
print("\nInitializing...")

import os
import numpy as np
import configparser
import argparse
from datetime import datetime

# Parse command-line parameters
ap = argparse.ArgumentParser()
ap.add_argument("--hyperband", "-H", action='store_true', help="Use Hyperband tuner", default=False)
args = vars(ap.parse_args())
use_hyperband = args["hyperband"]

# Parse parameters read from .ini file
config = configparser.ConfigParser()
config.read("train-tuner.ini")

DATASET_FILENAME = config.get("dataset", "DATASET_FILENAME")
NAME = DATASET_FILENAME[:-4] # Working subdirectory
X_VECTOR_SIZE = config.getint("dataset", "X_VECTOR_SIZE")
Y_VECTOR_SIZE = config.getint("dataset", "Y_VECTOR_SIZE")
VAL_SPLIT = config.getfloat("dataset", "VAL_SPLIT")

BATCH_SIZE = config.getint("model", "BATCH_SIZE")
LOSS = config.get("model", "LOSS")

DIRECTORY = config.get("tuner", "DIRECTORY")
RANDOM_SEED = config.getint("tuner", "RANDOM_SEED")
VERBOSITY = config.getint("tuner", "VERBOSITY")
BEST_MODELS = config.getint("tuner", "BEST_MODELS")
USE_GPU = config.getboolean("tuner", "USE_GPU")

if use_hyperband:
    from keras_tuner.tuners import Hyperband
    MAX_EPOCHS = config.getint("Hyperband", "MAX_EPOCHS")
    EARLY_STOPPING_PATIENCE = config.getint("Hyperband", "EARLY_STOPPING_PATIENCE")
else:
    from keras_tuner.tuners import RandomSearch
    EPOCHS = config.getint("RandomSearch", "EPOCHS")
    TRIALS = config.getint("RandomSearch", "TRIALS")
    EXECUTIONS_PER_TRIAL = config.getint("RandomSearch", "EXECUTIONS_PER_TRIAL")

# Set options and import TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
if not USE_GPU:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.keras.utils.set_random_seed(RANDOM_SEED)
    
# Load train dataset
dataset = np.loadtxt(DATASET_FILENAME, delimiter=",")

def build_model(hp):
    '''Model definition for Keras tuner'''
    model = tf.keras.models.Sequential()
    # Input and dense layer
    model.add(tf.keras.layers.Dense(
              input_dim=X_VECTOR_SIZE, units=hp.Int('units1', min_value=1, max_value=10, step=1),
              activation=hp.Choice('activation1', values=['tanh', 'sigmoid', 'relu'])))
    # Another dense layer
    #model.add(tf.keras.layers.Dense(
    #          units=hp.Int('units2', min_value=3, max_value=20, step=1),
    #          activation=hp.Choice('activation2', values=['tanh', 'sigmoid', 'relu'])))
    # Output layer
    model.add(tf.keras.layers.Dense(
              Y_VECTOR_SIZE, activation=hp.Choice('activation3', values=['tanh', 'sigmoid', 'relu'])))

    model.compile(loss=LOSS, optimizer=hp.Choice('optimizer', values=['adam', 'nadam', 'rmsprop']))
    return model

# Divide into predcitors and dependend variables
X = dataset[:, 0:X_VECTOR_SIZE]
Y = dataset[:, X_VECTOR_SIZE:]

# Set batch size as number of samples, if requested
if BATCH_SIZE == -1:
    BATCH_SIZE = len(X)
    config.set("model", "BATCH_SIZE", str(len(X)))

# Rescale Y, if needed
y_min = np.amin(Y, axis=0)
y_max = np.amax(Y, axis=0)
for j in range(Y_VECTOR_SIZE):
    if y_max[j] > 1 or y_min[j] < 0:
        Y[:,j] = (Y[:,j] - y_min[j]) / (y_max[j] - y_min[j])

# Rescale X, if needed
x_max = np.amax(X, axis=0)
x_min = np.amin(X, axis=0)
for j in range(X_VECTOR_SIZE):
    if x_max[j] > 1 or x_min[j] < 0:
        X[:,j] = (X[:,j] - x_min[j]) / (x_max[j] - x_min[j])

# Divide into training and validating sets
objective = "loss"
if VAL_SPLIT > 0:
    from sklearn.model_selection import train_test_split
    Xtrain, Xval, Ytrain, Yval = train_test_split(X, Y, test_size=VAL_SPLIT, random_state=RANDOM_SEED)
    objective = "val_loss"

# Tuner parameters
if use_hyperband:
    tuner = Hyperband(build_model, objective=objective, max_epochs=MAX_EPOCHS,
                      directory=DIRECTORY, project_name=NAME, seed=RANDOM_SEED)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor=objective, restore_best_weights=True,
                                                      patience=EARLY_STOPPING_PATIENCE)
else:
    tuner = RandomSearch(build_model, objective=objective, max_trials=TRIALS, project_name=NAME,
                         executions_per_trial=EXECUTIONS_PER_TRIAL, directory=DIRECTORY,
                         seed=RANDOM_SEED)

# Print some info
if VERBOSITY:
    tuner.search_space_summary()
print("\nSearch parameters:")
print("Dataset", config.items("dataset"))
print("Model  ", config.items("model"))
print("Tuner  ", config.items("tuner"))
if use_hyperband:
    print("Hyperband", config.items("Hyperband"))
else:
    print("RandomSearch", config.items("RandomSearch"))

# Run search
print("\nWorking...")
if VAL_SPLIT > 0:
    if use_hyperband:
        tuner.search(Xtrain, Ytrain, validation_data=(Xval, Yval), batch_size=BATCH_SIZE,
                     verbose=VERBOSITY, callbacks=[early_stopping])
    else:
        tuner.search(Xtrain, Ytrain, validation_data=(Xval, Yval), batch_size=BATCH_SIZE,
                     epochs=EPOCHS, verbose=VERBOSITY)
else:
    if use_hyperband:
        tuner.search(X, Y, batch_size=BATCH_SIZE, verbose=VERBOSITY, callbacks=[early_stopping])
    else:
        tuner.search(X, Y, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=VERBOSITY)

# Get best models / hyperparameters and print results
best_models = tuner.get_best_models(num_models=BEST_MODELS)
best_model_parameters = tuner.get_best_hyperparameters(num_trials=BEST_MODELS)
if VERBOSITY:
    tuner.results_summary(num_trials=BEST_MODELS)

outfilename = DIRECTORY+os.path.sep+NAME+"_"+str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))+".txt"
outfile = open(outfilename, "w")
outfile.write("Best model parameters:\n")
print("\nBest model parameters:")

for i,model in enumerate(best_models):

    outfile.write("\n\nModel no. {}\n".format(i))

    print(best_model_parameters[i].values)
    print(best_model_parameters[i].values, file=outfile)

    if VAL_SPLIT > 0:
        loss_t = model.evaluate(Xtrain, Ytrain, verbose=VERBOSITY)
        loss_v = model.evaluate(Xval, Yval, verbose=VERBOSITY)
        print("Score (train)      = {}".format(loss_t), file=outfile)
        print("Score (validation) = {}".format(loss_v), file=outfile)
    loss_c = model.evaluate(X, Y, verbose=VERBOSITY)
    print("Score (complete)   = {}".format(loss_c), file=outfile)
    print("Score (complete set) = {}\n".format(loss_c))

    model.summary(print_fn=lambda x: outfile.write(x + '\n'))

outfile.close()
print("Results saved to file:", outfilename)

print("Finished.")


