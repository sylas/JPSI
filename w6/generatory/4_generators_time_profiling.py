from timeit import default_timer as timer

# File for profile data output
fp = open('time_profile.log', 'w+')

MAX_RANGE = 10000000

def list_f():
    return [i for i in range(MAX_RANGE)]
    

def generator_f():
    return (i for i in range(MAX_RANGE))


if __name__ == '__main__':        

    # List part
    start_time = timer()
    my_list = list_f()
    elapsed_time = timer() - start_time
    fp.write("Creating list: " + str(timer() - elapsed_time)+" secs.\n")
    start_time = timer()
    for i in my_list:
        x = i*i
    elapsed_time = timer() - start_time
    fp.write("Iterating over list: " + str(elapsed_time)+" secs.\n")

    # Generator part
    start_time = timer()    
    my_generator = generator_f()
    elapsed_time = timer() - start_time
    fp.write("Creating generator: " + str(elapsed_time)+" secs.\n")
    start_time = timer()
    for i in my_generator:
        x = i*i
    elapsed_time = timer() - start_time
    fp.write("Iterating over generator: " + str(elapsed_time)+" secs.\n")

    fp.close()
