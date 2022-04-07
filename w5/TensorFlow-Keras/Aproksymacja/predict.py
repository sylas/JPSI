import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# CSV file name with test data
DATAFILE = "test.csv"

# Load test data from CSV file
X = np.loadtxt(DATAFILE, delimiter=",")

# Load model and print it's summary
model = load_model("model.h5")
model.summary()

# Calculate predictions
Ypredicted = model.predict(X)

# Plot results
xmin, xmax, ymin, ymax = plt.axis([-0.05, 1.05, 0, 0.8])
plt.plot(X, Ypredicted, ".")

# Add traing data to a plot
data = np.loadtxt('train.csv', delimiter=",")
Xt = data[:,0]
Yt = data[:,1]
plt.plot(Xt, Yt, "X")

plt.show()    
