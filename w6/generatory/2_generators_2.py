def generator_f():
    return (i for i in range(10))

my_generator = generator_f()

for i in my_generator:
    print(i, end=" ")

print()
# "Reset" generator
my_generator = generator_f()
for i in my_generator:
    print(i, end=" ")

print()
# "Reset" again
my_generator = generator_f()
# Print as a list
print(list(my_generator))

