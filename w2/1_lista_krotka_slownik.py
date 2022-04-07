# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# Lista - dostęp do elementów
list1 = ["a", 3.14, ["b","c"]]
print(list1)
print(list1[2])
print(list1[2][0])

# Lista - dodawanie i usuwanie elementów 
list1.append("Python")
try:
    list1.remove("a")
except ValueError:
    pass

# Lista - sortowanie
list2 = ["a","b","d","c","a"]
list2.sort()
print(list2)

# Lista - usuwanie elementów
del list2[0]
print(list2)


# Krotka
tuple = ("red", "green", "blue")
how_many_reds = tuple.count("red")
index_of_green = tuple.index("green")
print(how_many_reds, index_of_green)

tuple_single = (5, )


# Zbiór
set = {"a", "b", "d", "a","c"}
print(set)


# Słownik
translator = {"house": "dom", "praca": "work", 3.14: "pi"}
print(translator)
translator["school"] = "szkoła" # Nowa para danych
print(translator)
print(translator["praca"], translator[3.14])

dict_numbers = dict(number1=10, number2=15)
print(dict_numbers)

pracownik1={}
pracownik1["imie"] = "Janusz"
pracownik1["nazwisko"] = "Kowalski"
pracownik1["wiek"] = 30
pracownik2={"imie": "Adam", "nazwisko": "Nowak", "wiek": 35}
pracownicy = [pracownik1, pracownik2]
for pracownik in pracownicy:
    print(pracownik["nazwisko"])
pracownicy.remove(pracownik1)
print(pracownicy)

