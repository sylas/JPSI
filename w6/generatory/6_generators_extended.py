def generator_extended():
    i = 0
    while True:
        # Get the argument
        x = yield i 
        if x is None:
            i = i + 1
        else:
            i = x

my_generator = generator_extended()

print(next(my_generator), end=" ")
print(next(my_generator), end=" ")
print(next(my_generator), end=" ")

print(my_generator.send(8), end=" ")
print(next(my_generator), end=" ")
print(next(my_generator), end=" ")

print(my_generator.send(0), end=" ")
print(next(my_generator), end=" ")
print(next(my_generator), end=" ")


