import numpy as np

# Przypisanie tablicy do innej tablicy działa 
# jak referencja, a nie kopiowanie
a = np.array([1,2,3,4])
b = a
a[0] = 0
print(b)

# Kopiowanie tablicy
a = np.array([1,2,3,4])
b = np.copy(a)
a[0] = 0
print(b)

# Zapis / odczyt tablicy
a = np.random.random(5)
print(a)
np.save('saved_table', a) #  Dodane zostanie rozszerzenie .npy
b = np.load('saved_table.npy')
print(b)
print()

# Wczytanie pliku .csv
data = np.genfromtxt('summation.csv', delimiter=',', names=True)

# Zliczenie częstości występowania wybranych danych
data = np.array([0,1,1,1,2,2,2,2])
unique, counts = np.unique(data, return_counts=True)
print(unique)
print(counts)

# Sortowanie danych 
data = np.array([1,0,1,4,2,2,0,9,1])
print(sorted(data))