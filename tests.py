from main import *
import numpy as np
import time
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

def random(x):
    return x

def test_randomiser():
    rng = np.random.default_rng()
    with ProcessPoolExecutor(2) as executor:
        results = list(executor.map(random, rng.standard_normal(size=(10,10))))
    results1 = results[0]
    results2 = results[1]
    assert not np.allclose(results1, results2)
    print('Randomiser is thread-safe\n')

def test_k_space():

    start_time = time.time()

    k1, k2, k3, k4 = get_k_space()
    k5, k6, k7, k8 = get_k_space()
    assert np.allclose(k1, k5)
    assert np.allclose(k2, k6)
    assert np.allclose(k3, k7)
    assert np.allclose(k4, k8)
    print('>> k-space is reproducible')

    end_time = time.time()

    time_taken = np.round(end_time - start_time, 3)
    print(f"k-space-tests passed\nTime taken: {time_taken} seconds\n")

def test_hamiltonian():
    start_time = time.time()

    result = hamiltonian()

    assert np.allclose(result, result.conj().T)
    print('>> Hamiltonian is hermitian')
    assert np.allclose(1j*sy @ result.conj() @ (-1j*sy), result) # --> assertion fails
    print('>> Hamiltonian satisfies time-reversal symmetry')

    end_time = time.time()

    time_taken = np.round(end_time - start_time, 3)
    print(f"Hamiltonian tests passed\nTime taken: {time_taken} seconds\n")

def test_conductivity_vectorised_real_output(L=10):
    eta_factor = 1000
    N_i = 10
    R_I = np.random.uniform(low=-L/2, high=L/2, size=(2, N_i*L**2))
    u = 100
    l0 = L / 30

    start_time = time.time()

    conductivity = conductivity_vectorised(L, eta_factor, R_I, u, l0)
    assert np.allclose(conductivity.imag, 0, atol=1e-15)
    print('>> Conductivity is real')

    end_time = time.time()

    time_taken = np.round(end_time - start_time, 3)
    print(f"Conductivity tests passed\nTime taken: {time_taken} seconds")

def test_by_points():
    
    kx, ky, _, _ = get_k_space(L)

    
    def potential(kx, ky, R_i):
        gaus = u * np.exp(-(kx**2+ky**2)*l0**2/2)
        k_matrix = 0
        for i in range(N_i*int(L)**2):
            rands1 = R_i[0,i]
            rands2 = R_i[1,i]
            k_matrix += np.exp(1j * (kx * rands1 + ky * rands2)) * gaus
        return k_matrix/L**2


    def hamiltonian(kx, ky, R_i):
        sdotk = kx*sx2+ky*sy2
        v_q = potential(kx,ky,R_i)
        return 1*sdotk + v_q
    
    for i, j in zip(kx.ravel(), ky.ravel()):
        R_i = rng.uniform(low=-L/2,high=L/2,size=(2,N_i*L**2))
        h = hamiltonian(i, j, R_i)
        if np.allclose(1j*sy2 @ h.conj() @ (-1j*sy2), h):
            print(f'yes; h=\n{h}\nkx=\n{i}\nky=\n{j}')
    else:
        print('\n>>> k-points exhausted\n')

def conductivity():
    L_list = [9, l_max]
    N_i = 10
    strengths = [1]*3
    ranges = [l0]*3#,1e-3*l0, 1e-9*l0]
    configurations = 10
    for strength, Range in zip(strengths, ranges):
        plt.figure()
        conductivities = [[conductivity_vectorised(l,R_I=rng.uniform(low=-l/2,high=l/2,size=(2,N_i*l**2)),u=strength,l0=Range) for l in L_list] for _ in range(configurations)]
        plt.plot(L_list, np.mean(conductivities, axis=0))#, label=f'u={strength}, l0={range}')
        plt.title(f'$u={strength},l_0={Range}, configs={configurations}$')
        plt.xlabel('L')
        plt.ylabel('Conductivity')
        plt.tight_layout()
        name = determine_next_filename('Figure_',filetype='png',folder=r'C:\Users\freak\OneDrive\Desktop\cdoutputs')
        plt.savefig(name)
        plt.close()

if __name__ == '__main__':
    conductivity()
    # test_randomiser()
    # test_k_space()
    test_conductivity_vectorised_real_output()
    test_by_points()
    # test_hamiltonian()