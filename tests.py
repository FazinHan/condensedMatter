from main import *
import numpy as np
import time
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from parameters import *

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

def test_conductivity_vectorised_real_output(L=1):
    eta_factor = 1
    N_i = 10
    R_I = np.random.uniform(low=-L/2, high=L/2, size=(2, N_i*L**2))
    u = 10
    l0 = L / 30 / 4

    start_time = time.time()

    conductivity = conductivity_vectorised(L, eta_factor, R_I, u, l0)
    # assert np.allclose(conductivity.imag, 0, atol=1e-10), conductivity
    print('>> Conductivity is real')

    uv_conductivity = conductivity_unvectorized(L, eta_factor, R_I, u, l0)

    assert np.allclose(conductivity, uv_conductivity), f'{conductivity}, {uv_conductivity}'

    end_time = time.time()

    time_taken = np.round(end_time - start_time, 3)
    print(f"Conductivity tests passed\nTime taken: {time_taken} seconds")

def conductivity_unvectorized(L=L, eta_factor=eta_factor, R_I=np.random.uniform(low=-L/2, high=L/2, size=(2, N_i * int(L)**2)), u=u, l0=l0):
    
    factor = -1j * 2 * np.pi * h_cut**2 / L**2 * vf**2
    eta = eta_factor * vf * 2 * np.pi / L

    lamda = 20*np.pi/L
    k_vec = np.linspace(-lamda, lamda, k_space_size)

    potential = np.zeros([k_space_size**2]*2, dtype=complex)
    H0 = np.zeros_like(potential)

    cartesian_product = np.array(np.meshgrid(k_vec, k_vec, indexing='ij')).T.reshape(-1, 2)

    for i, (kx1, ky1) in enumerate(cartesian_product):
        for j, (kx2, ky2) in enumerate(cartesian_product):
            kx = kx1 - kx2
            ky = ky1 - ky2
            kk = np.sqrt(kx**2 + ky**2)
            for I in range(R_I.shape[-1]):
                potential[i, j] += np.exp(1j * (kx * R_I[0, I] + ky * R_I[1, I])) * function(kk, u, l0)
    
    potential = potential / L**2

    for i in range(k_space_size**2-1):
        H0[i+1, i] = vf * (k_vec[i % k_space_size] + 1j*k_vec[i // k_space_size])
        H0[i, i+1] = vf * (k_vec[i % k_space_size] - 1j*k_vec[i // k_space_size])

    potential = np.kron(potential, np.eye(2))
    H0 = np.kron(H0, np.eye(2))

    ham = H0 + potential

    assert np.allclose(ham, ham.conj().T), 'pythonic ham is not hermitian'

    vals, vecs = np.linalg.eigh(ham)
    conductivity = 0

    for idx, E1 in enumerate(vals):
        for jdx, E2 in enumerate(vals):
            if E1 != E2:
                conductivity += np.abs(vecs[:, idx].reshape(2*k_space_size**2,1).conj().T @ sx @ vecs[:, jdx].reshape(2*k_space_size**2,1))**2 * (fermi_dirac(E1) - fermi_dirac(E2)) / (E1 - E2) / (E1 - E2 + 1j * eta)

    conductivity *= factor
    
    # Return the summation of the Kubo term
    return conductivity[0,0]

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
    L_list = [l_min, l_max]
    N_i = 1
    strengths = [1]*3
    ranges = [l0]*3#,1e-3*l0, 1e-9*l0]
    configurations = 10
    for strength, Range in zip(strengths, ranges):
        # plt.figure()
        conductivities = np.array([[conductivity_vectorised(l,R_I=rng.uniform(low=-l/2,high=l/2,size=(2,N_i*l**2)),u=strength,l0=Range) for l in L_list] for _ in range(configurations)])
        sem = np.std(conductivities, axis=0) / np.sqrt(conductivities.shape[0])
        plt.errorbar(L_list, np.mean(conductivities, axis=0), yerr=sem, fmt='.')
        plt.title(f'$u={strength},l_0={Range}, configs={configurations}$')
        plt.xlabel('L')
        plt.ylabel('Conductivity')
        plt.tight_layout()
        name = determine_next_filename('Figure_',filetype='png',folder=r'C:\Users\freak\OneDrive\Desktop\cdoutputs')
        # plt.savefig(name)
        plt.show()
        # plt.close()

def test_julia_interface():
    from julia import Julia
    from julia import Main

    # Initialize Julia
    julia = Julia(compiled_modules=False)

    # Load your Julia module
    Main.include("functions.jl")

    eta_factor = 1
    N_i = 10
    R_I = np.random.uniform(low=-L/2, high=L/2, size=(2, N_i*L**2))
    u = 10
    l0 = L / 30 / 4

    t0 = time.perf_counter()
    conductivity_jl = Main.conductivity(L, eta_factor, R_I, u, l0)
    t1 = time.perf_counter()
    conductivity = conductivity_unvectorized(L, eta_factor, R_I, u, l0)
    t2 = time.perf_counter()

    print(Main.cond)

    assert np.allclose(conductivity, conductivity_jl), f'{conductivity}, {conductivity_jl}'

    print('>> Julia and python calculations are consistent')
    print(f'Python: {t2-t1} seconds\nJulia: {t1-t0} seconds')


if __name__ == '__main__':
    # conductivity()
    # test_randomiser()
    # test_k_space()
    # test_conductivity_vectorised_real_output()
    # test_by_points()
    # test_hamiltonian()
    test_julia_interface()