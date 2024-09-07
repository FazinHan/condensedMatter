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
    assert np.allclose(k1+k1.T, np.zeros_like(k1))
    assert np.allclose(k2+k2.T, np.zeros_like(k2))
    print('>> k-space is anti-symmetric')

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

    assert np.allclose(conductivity, conductivity_unvectorized(L, eta_factor, R_I, u, l0))

    end_time = time.time()

    time_taken = np.round(end_time - start_time, 3)
    print(f"Conductivity tests passed\nTime taken: {time_taken} seconds")

def conductivity_unvectorized(L=L, eta_factor=eta_factor, R_I=None, u=u, l0=l0):
    if R_I is None:
        R_I = np.random.uniform(low=-L/2, high=L/2, size=(2, N_i * int(L)**2))
    
    factor = -1j * 2 * np.pi * h_cut**2 / L**2 * vf**2
    eta = eta_factor * vf * 2 * np.pi / L
    
    if L == l_min:
        start_time = time.time()
        ham = hamiltonian(L, R_I, u, l0)
        end_time = time.time()
        vals, vecs = np.linalg.eigh(ham)
        diag_time = time.time()
        execution_time = np.round(end_time - start_time, 3)
        diag_time = np.round(diag_time - end_time, 3)
        print(f"\nHamiltonian computed in: {execution_time} seconds\nDiagonalised in {diag_time} seconds\n")
    else:
        ham = hamiltonian(L, R_I, u, l0)
        vals, vecs = np.linalg.eigh(ham)
    
    sx_vecs = sx@vecs
    vecs_conj = np.conjugate(vecs)
    
    E0 = []
    E1 = []
    
    # Manually create the meshgrid for E0 and E1
    for i in range(len(vals)):
        for j in range(len(vals)):
            E0.append(vals[i])
            E1.append(vals[j])
    
    fd_diff = []
    
    # Compute the Fermi-Dirac difference manually
    for i in range(len(E0)):
        fd_diff.append(fermi_dirac_ondist(E0[i]) - fermi_dirac_ondist(E1[i]))
    
    diff = []
    
    # Compute the difference manually and add a small value to avoid division by zero
    for i in range(len(E0)):
        diff.append(E0[i] - E1[i] + 1e-10)
    
    n_prime_dagger_sx_n_mod2 = np.zeros_like(ham, dtype=complex)
    
    # Compute the summation for n_prime_dagger_sx_n_mod2
    for i in range(len(vecs)):
        for j in range(len(vecs)):
            n_prime_dagger_sx_n_mod2[i, j] = 0
            for k in range(len(vecs)):
                n_prime_dagger_sx_n_mod2[i, j] += np.abs(vecs_conj[i, k] * sx_vecs[j, k])**2
    
    kubo_term = np.zeros_like(ham, dtype=complex)
    
    # Compute the Kubo term
    for i in range(len(diff)):
        kubo_term[i // len(vals), i % len(vals)] = factor * fd_diff[i] / diff[i] * n_prime_dagger_sx_n_mod2[i // len(vals), i % len(vals)] / (diff[i] + 1j * eta)
    
    # Return the summation of the Kubo term
    return np.sum(kubo_term)

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
    N_i = 1
    strengths = [1]*3
    ranges = [l0]*3#,1e-3*l0, 1e-9*l0]
    configurations = 10
    for strength, Range in zip(strengths, ranges):
        # plt.figure()
        conductivities = [[conductivity_vectorised(l,R_I=rng.uniform(low=-l/2,high=l/2,size=(2,N_i*l**2)),u=strength,l0=Range) for l in L_list] for _ in range(configurations)]
        plt.plot(L_list, np.mean(conductivities, axis=0))#, label=f'u={strength}, l0={range}')
        plt.title(f'$u={strength},l_0={Range}, configs={configurations}$')
        plt.xlabel('L')
        plt.ylabel('Conductivity')
        plt.tight_layout()
        name = determine_next_filename('Figure_',filetype='png',folder=r'C:\Users\freak\OneDrive\Desktop\cdoutputs')
        plt.savefig(name)
        # plt.close()

if __name__ == '__main__':
    # conductivity()
    # test_randomiser()
    test_k_space()
    test_conductivity_vectorised_real_output()
    # test_by_points()
    # test_hamiltonian()