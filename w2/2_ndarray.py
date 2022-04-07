import numpy as np

# Przykładowa tablica
na = np.array([[1, 3.14],[0.5, 1.2]])
print(na, na.dtype)

# Wyświetlenie właściwości tablicy
print(na.ndim, na.size, na.shape, sep=", ")

# Zmiana wymiaru
nb = na.reshape(1,4)

# Alternatywnie: reshape((1,4)) lub np.reshape(na, (1,4))
print(nb)
print(na.reshape(4))

# Tworzenie tablicy
ad = np.array([[1, 2, 3],[4, 5, 6]], dtype=np.float32)
print(ad)

# Wykorzystaniem funkcji inicjujących zawartość
a0 = np.zeros((2,2))
a1 = np.ones((2,2), dtype=np.int16)
print(a0)
print(a1)

ar = np.arange(1,9,2)
print(ar)
print(ar.reshape(2,2))

# Wartości losowe
rand = np.random.random(5)
print(rand)

# Podanie zakresu dla tabeli jednowymiarowej
a = np.arange(4)
print(a)
b = np.arange(4,8)
print(b)
print(a+b)
print(a+4)

# Tabela dwuwymiarowa
c = np.arange(9).reshape(3,3)
print(np.sin(c))
print(c*c)
print(np.dot(c,c))

print(c[1,2], c[1,:], c[:,1])

# Funkcje uniwersalne
e = [1,2,3]
print(np.sqrt(e), np.log(e))

# Funkcje agregujące
print(np.sum(e),np.mean(e))

# Indeksowanie elementów tablicy
print(e[1],e[-1],e[0:1],e[:-1])


# Iterowanie po elementach tablicy
for i in c:
    print(i)
    
for i in c.flat:
    print(i)

m_c = np.apply_along_axis(np.mean, axis=0, arr=c)
m_r = np.apply_along_axis(np.mean, axis=1, arr=c)
print(m_c)
print(m_r)

# Osie
a = np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]], dtype=np.float32)
print(np.sum(a))
print(np.sum(a, axis=0))
print(np.sum(a, axis=1))
