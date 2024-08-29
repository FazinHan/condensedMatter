from mpi4py import MPI; rank = MPI.COMM_WORLD.Get_rank()

#os.mkdir(os.path.join('output_data','results_version','run'+rank))

print(rank)
exit()

import numpy as np
from concurrent.futures import ProcessPoolExecutor
import os, warnings, sys, time

l_min, l_max = 10,40
num_lengths = 15

vf = 1e3
h_cut = 1
u = 1
l0 = l_min / 30
N_i = 10
L = 10
# l0 = L/30
eta_factor = 1e3
T = 0
ef = 0
# a = 1

configurations = 1
k_space_size = 20

interaction_distance = 3

rng = np.random.default_rng()

sx2 = np.array([[0,1],[1,0]])
sy2 = 1j * np.array([[0,-1],[1,0]])
sz2 = np.eye(2)
sz2[-1,-1] = -1

def gaussian_corr(q, u, l0):
    return u * np.exp(-q**2*l0**2/2)

def thomas_fermi(q, u, l0):
    return u / (q + l0**-1)

# try:
#     sys.argv[2]
#     function = thomas_fermi
# except IndexError:
#     function = gaussian_corr

function = gaussian_corr
    
def get_k_space(L=L):
    '''
    k_space_size = 51
    161 μs ± 5.12 μs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
    '''
    lamda = 20*np.pi/L
    k_vec = np.linspace(-lamda, lamda, k_space_size) # N
    
    cartesian_product = np.array(np.meshgrid(k_vec, k_vec, indexing='ij')).T.reshape(-1, 2)
       
    k1x, k2x = np.meshgrid(cartesian_product[:,0], cartesian_product[:,0],sparse=True) # N^2
    k1y, k2y = np.meshgrid(cartesian_product[:,1], cartesian_product[:,1],sparse=True)

    kx = k1x - k2x
    ky = k1y - k2y

    # k1x, k1y = np.meshgrid(k_vec, k_vec) # N, N
    
    return kx, ky.T, np.diag(cartesian_product[:,0]), np.diag(cartesian_product[:,1])

_, _, kx, _ = get_k_space(L)
sx = np.kron(sx2, np.eye(kx.shape[0]))
sy = np.kron(sy2, np.eye(kx.shape[0]))
sz = np.kron(sz2, np.eye(kx.shape[0]))

def ft_potential_builder_3(L=L, R_I=rng.uniform(low=-L/2,high=L/2,size=(2,N_i*L**2)), u=u, l0=l0):
    '''
    2min 2s ± 1.97 s per loop (mean ± std. dev. of 7 runs, 1 loop each)
    '''
    kx, ky, _, _ = get_k_space(L)
    
    k_matrix = np.zeros_like(kx, dtype=np.complex128)
    
    for i in range(N_i*int(L)**2):
        rands1 = np.ones_like(kx)*R_I[0,i]
        rands2 = np.ones_like(ky)*R_I[1,i] # may not be needed
        k_matrix += np.exp(1j * (kx * rands1 + ky * rands2)) * function( (kx**2 + ky**2)**.5 , u, l0) 

    return np.kron(np.eye(2), k_matrix)/L**2

def fermi_dirac_ondist(x,T=T,ef=ef): # T=1e7*vf*2*np.pi
    if T != 0:
        return 1/(1+np.exp((x-ef)/T))
    out = np.zeros(x.shape)
    out[np.where(x < ef)] = 1
    out[np.where(x == ef)] = .5
    return out

def fermi_dirac(x,T=T,ef=ef):
    if T != 0:
        return 1/(1+np.exp((x-ef)/T))
    if x>ef:
        return 0
    if x==ef:
        return .5
    return 1

def hamiltonian(L=L, R_I=rng.uniform(low=-L/2,high=L/2,size=(2,N_i*L**2)), u=u, l0=l0):
    '''1min 27s ± 3.7 s per loop (mean ± std. dev. of 7 runs, 1 loop each)'''
    _, _, kx, ky = get_k_space(L)

    sdotk = np.kron(sx2, kx) + np.kron(sy2, ky)
    H0 = vf * sdotk
    V_q = ft_potential_builder_3(L, R_I, u, l0)
    return H0 + V_q

def conductivity_vectorised(L=L, eta_factor=eta_factor, R_I=rng.uniform(low=-L/2,high=L/2,size=(2,N_i*L**2)), u=u, l0=l0): # possibly the slowest function
    '''
    999.97s on default function call: k_space_size = 45
    
    31.5 s ± 364 ms per loop (mean ± std. dev. of 7 runs, 1 loop each): k_space_size = 20
    '''
    factor = -1j * 2 * np.pi * h_cut**2/L**2 * vf**2

    eta = eta_factor * vf * 2 * np.pi / L

    
    if L == l_min:
        start_time = time.time()
        ham = hamiltonian(L, R_I, u, l0)
        end_time = time.time()
        vals, vecs = np.linalg.eigh(ham) 
        diag_time = time.time()
        execution_time = np.round(end_time - start_time, 3)
        diag_time = np.round(diag_time - end_time, 3)
        print(f"Hamiltonian computed in: {execution_time} seconds\nDiagonalised in {diag_time} seconds")
        assert np.allclose(ham.T.conj(), ham)
        # assert np.allclose(sz @ ham @ (-sz), ham) # --> assertion error
        assert np.allclose(1j*sy @ ham.conj() @ (-1j*sy), ham) # --> assertion error
    else:
        ham = hamiltonian(L, R_I, u, l0)
        vals, vecs = np.linalg.eigh(ham) 

    sx_vecs = sx @ vecs # i checked this, matrix multiplication is equivalent to multiplying vector-wise 
    vecs_conj = vecs.conj()

    E0, E1 = np.meshgrid(vals, vals)
    fd_diff = fermi_dirac_ondist(E0) - fermi_dirac_ondist(E1)
    diff = E0-E1+1e-10

    n_prime_dagger_sx_n_mod2 = np.zeros_like(ham)
    for i in range(vecs.shape[0]):
        n_dagger, sx_n = np.meshgrid(vecs_conj[i,:],sx_vecs[i,:],sparse=True)
        n_prime_dagger_sx_n_mod2 += np.abs(n_dagger * sx_n)**2

    kubo_term = factor * fd_diff / diff * n_prime_dagger_sx_n_mod2 / (diff + 1j*eta)

    # print(kubo_term)
    
    return np.sum(kubo_term)

def main(L=np.linspace(l_min,l_max,num_lengths)): # faster locally (single node)

    print('main function run')

    start_time = time.time()
    conductivities = np.array([conductivity_vectorised(l, eta_factor, rng.uniform(low=-l/2,high=l/2,size=(2,N_i*int(l)**2)), u, l0) for l in L])
    end_time = time.time()
    execution_time = np.round(end_time - start_time, 3)

    print(f'conductivities computed in {execution_time} seconds')

    conductivities = str(conductivities.tolist())

    dirname = os.path.join('output_data','results_version','run'+rank)
    fname = determine_next_filename(fname='output',folder=dirname, filetype='txt')
    write_file(fname, conductivities)
    print('conductivities stored')

def write_file(fname, array):
    with open(fname, 'w') as file:
        file.write(array)

def determine_next_filename(fname='output',filetype='png',folder='graphics',direc=False,exists=False):
    num = 1
    if direc:
        filename = lambda num: f'{fname}{num}'
        while os.path.isdir(os.path.join('.',folder,filename(num))):
            num += 1
        else:
            if exists:
                num -= 1
        return os.path.join(folder,filename(num))
    filename = lambda num: f'{fname}{num}.{filetype}'
    while os.path.isfile(os.path.join('.',folder,filename(num))):
        num += 1
    else:
        if exists:
            num -= 1
    return os.path.join(folder,filename(num))

import mpi4py; rank = mpi4py.MPI.Comm().Get_rank()

os.mkdir(os.path.join('output_data','results_version','run'+rank))

if __name__ == "__main__":
    
    L = [np.linspace(l_min, l_max,num_lengths)] * configurations

    with ProcessPoolExecutor() as exe:
        _ = exe.map(main, L)

    dirname = os.path.join('output_data','results_version','run'+rank)
    
    fname = os.path.join('output_data','results_version','length.npy')

    if not os.path.isfile(fname):
        np.save(fname, L[0])
    if not os.path.isfile(os.path.join('output_data','results_version','params.txt')):
        with open(os.path.join('output_data','results_version','params.txt'),'w') as file:
            text = f'''l_min, l_max = {l_min}, {l_max}\neta = {eta_factor} * vf * 2 * pi / L\nvf = {vf}\nh_cut = {h_cut}\nu = {u}\nl0 = {l0}\nN_i = {N_i}\nT = {T}\nef = {ef}\nconfigurations = {configurations}\nk_space_size = {k_space_size}\nscattering potential = {function}'''#\na = {a}'''
            file.write(text)
            print('parameter file written')
