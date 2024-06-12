from concurrent.futures import ProcessPoolExecutor
import numpy as np

with open('function_outputs\\massless.npy','rb') as file:
    values = np.load(file)
    vectors = np.load(file)
    values_plot = np.load(file)

sx = np.array([[0,1],[1,0]])

def nsn2_func(vecs):
    result = []
    for i in range(vecs.size):
        for j in range(vecs[i:].size)
            result.append(np.abs(vecs[i].conj().T @ sx @ vecs[j])**2)
    return np.array(result)

def enn_func(vals):
    result = []
    for i in range(vecs.size):
        for j in range(vecs[i:].size)
            result.append(vals[i] - vals[j])
    return np.array(result)

def main():
    nsn2 = []
    enn_ = []
    with ProcessPoolExecutor(20) as executor:
        nsn2_vals = executor.map(nsn2_func, vectors)
        enn_vals = executor.map(enn_func, values)
        for j,k in zip(nsn2_vals,enn_vals):
            nsn2.append(j)
            enn_.append(k)
        print(f"appended for value {i}\n")

if __name__ == '__main__':
    main()
            