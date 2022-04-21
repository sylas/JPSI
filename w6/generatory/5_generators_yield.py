def generator_f():
    for i in range(10):
        yield i

my_generator = generator_f()

for i in my_generator:
    print(i, end=" ")



def generator_infinite():
    i = 0
    while True:
        yield i
        i = i + 1

my_generator = generator_infinite()

for _ in range(10):
    print(next(my_generator), end=" ")



def generator_infinite_stop():
    i = 0
    while True:
        if i == 5:
            raise StopIteration
        else:
            yield i
            i = i + 1

my_generator = generator_infinite_stop()

# Raises StopIteration 
for _ in range(10):
    print(next(my_generator), end=" ")
