import time
import ray

# Initialize Ray

n = 10
@ray.remote
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
start = time.time()

def runner():
    # List to hold the future objects returned by the remote function
    futures = []
    # Launch the tasks in parallel
    for i in range(n):
        j = 500000 + i
    futures.append(fib.remote(j))

    # Gather the results
    results = ray.get(futures)

    # Create the return dictionary
    return_dict = {i: result for i, result in enumerate(results)}
    # Shutdown Ray


    return "Hola"


