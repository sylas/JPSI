import pandas as pd
import numpy as np

# Series - szereg
s = pd.Series([3,2,9,9])
print(s)

# Własne indeksy
s = pd.Series([3,2,9,9], index=['a','b','c',"d"])

# Filtrowanie wartości
print(s[s>5])

# Zliczanie wartości unikalnych
print(s.unique())
print(s.value_counts())

#Sprawdzenie duplikatów
print(s.duplicated())


# DataFrame
# Tworzenie na podstawie słownika
data = {"name": ["Jan", "Andrzej", "Jerzy"],
        "surname": ["Kowalski", "Nowak", "Iksiński"],
        "age": [30, 35, 40]}
frame = pd.DataFrame(data)
print(frame)

# Tylko wybrane kolumny ze słownika
frame2 = pd.DataFrame(data, columns=["name","surname"])
print(frame2)

# Wczytanie z pliku .csv
csvframe = pd.read_csv('summation.csv')
print(csvframe)

# Podstawowe operacje
data = {"name": ["Jan", "Andrzej", "Jerzy"],
        "surname": ["Kowalski", "Nowak", "Iksiński"],
        "age": [30, 35, 40]}
frame = pd.DataFrame(data)

print(frame.columns)

print(frame.index)

print(frame.values)

print(frame.surname) # Albo: frame["surname"] 

print(frame[0:2])

# Nowa kolumna na podstawie listy
frame["salary"] = [5000, 4500, 6200]
print(frame)

# Nowa kolumna na podstawie Series
s = pd.Series([0, 2, 1])
frame["children"] = s
print(frame)

#Filtrowanie danych
print(frame[frame.children > 0])

# Statystyki danych
print(frame.describe())

# Transpozycja
print(frame.T)

# Usunięcie kolumny
del frame["age"]
print(frame)

# Zmiana nazwy kolumny („w miejscu”)
frame.rename(columns=lambda x: x.upper(), inplace=True)
print(frame)

