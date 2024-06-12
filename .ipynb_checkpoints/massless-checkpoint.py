from concurrent.futures import ProcessPoolExecutor
import numpy as np

with open('function_outputs\\massless.npy','rb') as file:
    values = np.load(file)
    vectors = np.load(file)
    values_plot = np.load(file)

sx = np.array([[0,1],[1,0]])

def nsn2_func(vec1,vec2):
    return np.abs(vec1.conj().T @ sx @ vec2)**2

def enn_func(val1, val2):
    return val1 - val2

def main():
    nsn2 = []
    enn_ = []
    with ProcessPoolExecutor(20) as executor:
        for i in range(values.size):
            nsn2_vals = executor.map(lambda p: nsn2_func(vectors[i],p), vectors[i:])
            enn_vals = executor.map(lambda p: enn_func(values[i],p), values[i:])
            for j,k in zip(nsn2_vals,enn_vals):
                nsn2.append(j)
                enn_.append(k)
            print(f"appended for value {i}\n")

if __name__ == '__main__':
    main()
            