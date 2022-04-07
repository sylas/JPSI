from sklearn import preprocessing
import numpy as np

X_train = np.array([[1., -2.,  3.], [ 2., 0., -1.], [ 0.,  1., 100.]])
print("Dane oryginalne:")
print(X_train)


Q1 = np.percentile(X_train, 25)
Q3 = np.percentile(X_train, 75)
IQR = Q3 - Q1
x_min_range = Q1 - 1.5*IQR
x_max_range = Q3 + 1.5*IQR

for i,d in np.ndenumerate(X_train):
    if not x_min_range < d < x_max_range:
        print("Outlier:", i, d)
        X_train[i] = np.median(X_train)

print("Dane po usunięciu wartości odstających:")
print(X_train)

