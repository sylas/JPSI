# Installation: pip install memory-profiler
from memory_profiler import profile

# File for profile data output
fp = open('memory_profile.log', 'w+')

MAX_RANGE = 10000000

@profile(stream=fp)
def list_f():
    return [i for i in range(MAX_RANGE)]    

@profile(stream=fp)
def generator_f():
    return (i for i in range(MAX_RANGE))


if __name__ == '__main__':        

    # List part
    my_list = list_f()
    for i in my_list:
        x = i*i

    # Generator part
    my_generator = generator_f()
    for i in my_generator:
        x = i*i

    fp.close()

