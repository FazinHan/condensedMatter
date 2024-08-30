from main import *
import numpy as np
import time
import matplotlib.pyplot as plt

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

    end_time = time.time()

    time_taken = np.round(end_time - start_time, 3)
    print(f"Conductivity tests passed\nTime taken: {time_taken} seconds")

if __name__ == '__main__':
    test_randomiser()
    test_k_space()
    test_conductivity_vectorised_real_output()
    test_hamiltonian()