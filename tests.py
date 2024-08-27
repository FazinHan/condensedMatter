from main import *
import numpy as np
import time
import matplotlib.pyplot as plt

def test_hamiltonian():
    start_time = time.time()

    result = hamiltonian()

    # plt.matshow((sz @ result @ (-sz)) == result)
    # plt.colorbar()
    # plt.savefig(os.path.join('outtests','szhsz_h.png'))
    plt.matshow((1j*sy @ result.conj() @ (-1j*sy)) == result)
    plt.colorbar()
    # plt.show()
    plt.savefig(os.path.join('outtests','isyhisy_h.png'))

    assert np.allclose(result, result.conj().T)
    assert np.allclose(1j*sy @ result.conj().T @ (-1j*sy), result) # --> assertion error
    # assert np.allclose(sz @ result @ (-sz), result)

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