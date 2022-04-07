from sklearn import preprocessing
import numpy as np

# Dane wejściowe
X_train = np.array([[1., -2.,  3.], [ 2., 0., -1.], [ 0.,  1., -2.]])
print("Dane oryginalne:")
print(X_train)

# Obliczenie funkcji skalującej i wyświetlenie jej właściwości
scaler = preprocessing.StandardScaler().fit(X_train)
print("Średnia:", scaler.mean_)
print("Wariancja:", X_train.std(axis=0)**2)

# Transformacja zbioru
X_scaled = scaler.transform(X_train)
print("Tablica przeskalowana do średniej = 0 i wariancji = 1:")
print(X_scaled)

# Sprawdzenie statystyk po transformacji
print("Średnia:", X_scaled.mean(axis=0))
print("Wariancja:", X_scaled.std(axis=0)**2)

