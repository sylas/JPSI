from sklearn.model_selection import train_test_split
import numpy as np

# Losowa tablica
print("Generuję zbiór danych - 100 liczb losowych...")
data = np.random.random(100)

# Ułamek danych, który będzie danymi testowymi
SPLIT = 0.2

# Ziarno generatora liczb pseudolosowych
RANDOM_SEED = 100

print("Dzielę na zbiór trenujący i walidujący...")
(train_data, test_data) = train_test_split(data, 
      test_size=SPLIT, random_state=RANDOM_SEED)

print("Dane trenujące ({} elementów):".format(len(train_data)))
print(train_data)

print("Dane walidujące ({} elementów):".format(len(test_data)))
print(test_data)
