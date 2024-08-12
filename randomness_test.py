import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from time import sleep

'''
taken from https://stackoverflow.com/questions/49847794/child-processes-generating-same-random-numbers-as-parent-process
op sees less randomness, unlike here
'''

rng = np.random.default_rng()

def calc_g():
    sleep(1)
    u = rng.uniform()
    u1 = rng.uniform()
    print(u, u1)

futures = {}

if __name__ == "__main__":
    with ProcessPoolExecutor(14) as executor:
    
        for i in range(14):  
            job = executor.submit(calc_g)
            futures[job] = i
    
        for job in as_completed(futures):
            job.result()