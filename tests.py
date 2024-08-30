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
    print('>> Conductivity is real')

    end_time = time.time()

    time_taken = np.round(end_time - start_time, 3)
    print(f"Conductivity tests passed\nTime taken: {time_taken} seconds")

def test_by_points():
    
    lamda = 20*np.pi/L
    k_vec = np.linspace(-lamda, lamda, k_space_size) # N
    cartesian_product = np.array(np.meshgrid(k_vec, k_vec, indexing='ij')).T.reshape(-1, 2)
    
    def potential(kx, ky):
        gaus = u * np.exp(-(kx**2+ky**2)*l0**2/2)
        R_i = rng.uniform(low=-L/2,high=L/2,size=(2,N_i*L**2))
        k_matrix = 0
        for i in range(N_i*int(L)**2):
            rands1 = R_i[0,i]
            rands2 = R_i[1,i]
            k_matrix += np.exp(1j * (kx * rands1 + ky * rands2)) * gaus
        return k_matrix/L**2


    def hamiltonian(kx, ky):
        sdotk = kx*sx+ky*sy
        v_q = potential(kx,ky)
        return 1*sdotk + v_q
    
    for i, j in cartesian_product:
        h = hamiltonian(i, j)
        if np.allclose(1j*sy @ h.conj().T @ (-1j*sy), h):
            print(f'yes; h=\n{h}\nkx=\n{i}\nky=\n{j}')
    else:
        print('>>> k-points exhausted\n')

if __name__ == '__main__':
    test_randomiser()
    test_k_space()
    test_conductivity_vectorised_real_output()
    test_by_points()
    test_hamiltonian()