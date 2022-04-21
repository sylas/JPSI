# Create list and display
my_list = [i for i in range(10)]
print(my_list)

# Create generator and try to display
my_generator = (i for i in range(10))
print(my_generator)

# Display subsequent items
print(next(my_generator))
print(next(my_generator))
print(next(my_generator))

# Loop over generator 
for i in my_generator:
    print(i)

# Catch the exception  
try:
    print(next(my_generator))
except StopIteration:
    print("No more elements in generator")

