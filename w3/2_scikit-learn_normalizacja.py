from sklearn import preprocessing
import numpy as np

# Dane wejściowe
X_train = np.array([[1., -2.,  3.], [ 2., 0., -1.], [ 0.,  1., -2.]])
print("Dane oryginalne:")
print(X_train)

min_max_scaler = preprocessing.MinMaxScaler().fit(X_train)
X_scaled_0_1 = min_max_scaler.transform(X_train)
print("Tablica przeskalowana do przedziału [0, 1]:")
print(X_scaled_0_1)

# Normalizacja danych rzadkich
X_train = np.array([[0., 2.,  0.], [ 0., 0., 0.], [ 0.,  0, -3.]])
print("Dane oryginalne (rzadkie):")
print(X_train)

# Fit i transform "w jednym"
# Wykorzystanie MaxAbsScaler - zachowuje rzadkość danych
abs_scaler = preprocessing.MaxAbsScaler().fit(X_train)
print("Tablica przeskalowana do przedziału [0, 1]:")
print(abs_scaler.transform(X_train))
