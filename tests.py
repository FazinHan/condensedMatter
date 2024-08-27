from main import *
import numpy as np
import time

def test_hamiltonian():
    sy = np.array([[0,-1j],[1j,0]])
    _, _, _, ky = get_k_space(L)
    sy = np.kron(sy, ky)

    start_time = time.time()

    result = hamiltonian()

    assert np.allclose(result, result.conj().T)
    assert np.allclose(sy @ result.conj() @ sy, result)

    end_time = time.time()

    time_taken = np.round(end_time - start_time, 3)
    print(f"Hamiltonian tests passed\nTime taken: {time_taken} seconds")

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
    test_hamiltonian()
    test_conductivity_vectorised_real_output()