from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy as np

# Dane treningowe
X = [[1, 1, 1],
     [1, 1, 2],
     [1, 2, 2],
     [2, 1, 1],
     [8, 8, 8],
     [7, 8, 8],     
     [7, 7, 8],
     [9, 8, 7],
     [1, 2, 1],
     [2, 2, 1]]

# Etykiety danych
Y = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0]

print("Dane treningowe i ich etykiety:")
for i in range(len(X)):
     print(X[i],Y[i])

# Dzielimy na zbiór treningowy (80%) i testowy (20%)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Klasyfikator SVN i jego trenowanie
model = svm.SVC()
model.fit(X_train, Y_train)

# Predykcja na zbiorze testowym (walidacyjnym)
res = model.predict(X_test)

print("Predykcja na zbiorze walidującym:")
# Wydruk wyników (dana - klasyfikacja)
for i in res:
    print(X_test[i], "->", i)

# Predykcja na kolejnej danej
data = [[1,1,0]]
print("Predykcja dla danej testowej {} -> {}".format(data[0], model.predict(data)[0]))
