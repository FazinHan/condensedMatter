import numpy as np
from scipy import stats
from scipy.fft import fft2, rfft2
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import os, warnings, sys, time

'''implement a check on the hamiltoninan: σz(H)σz = −(H) '''

l_min, l_max = 10,40
num_lengths = 15

vf = 1e3
h_cut = 1
u = 1
l0 = l_min / 30
N_i = 10
L = 10
# l0 = L/30
eta_factor = 1
T = 0
ef = 0
# a = 1

configurations = 1
k_space_size = 20

interaction_distance = 3

rng = np.random.default_rng()

sx = np.array([[0,1],[1,0]])
sy = 1j * np.array([[0,-1],[1,0]])

def gaussian_corr(q):
    return u * np.exp(-q**2*l0**2/2)

def thomas_fermi(q):
    return u / (q + l0**-1)

try:
    #sys.argv[2]
    function = thomas_fermi
except IndexError:
    function = gaussian_corr
    
def get_k_space(L=L):
    '''
    k_space_size = 51
    161 μs ± 5.12 μs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
    '''
    lamda = 20*np.pi/L
    k_vec = np.linspace(-lamda, lamda, k_space_size) # N
    
    cartesian_product = np.array(np.meshgrid(k_vec, k_vec, indexing='ij')).T.reshape(-1, 2)
    
    k1x, k2x = np.meshgrid(cartesian_product[:,0], cartesian_product[:,0]) # N^2
    k1y, k2y = np.meshgrid(cartesian_product[:,1], cartesian_product[:,1])

    kx = k1x - k2x
    ky = k1y - k2y

    # k1x, k1y = np.meshgrid(k_vec, k_vec) # N, N
    
    return kx, ky, np.diag(cartesian_product[:,0]), np.diag(cartesian_product[:,1])

def ft_potential_builder_3(L=L, R_I=rng.uniform(low=-L/2,high=L/2,size=(2,N_i*L**2))):
    '''
    2min 2s ± 1.97 s per loop (mean ± std. dev. of 7 runs, 1 loop each)
    '''
    kx, ky, _, _ = get_k_space(L)
    
    k_matrix = np.zeros_like(kx, dtype=np.complex128)
    
    for i in range(N_i*int(L)**2):
        rands1 = np.ones_like(kx)*R_I[0,i]
        rands2 = np.ones_like(ky)*R_I[1,i] # may not be needed
        k_matrix += np.exp(1j * (kx * rands1 + ky * rands2)) * function( (kx**2 + ky**2)**.5 ) 

    return np.kron(k_matrix,np.eye(2))/L**2

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

def hamiltonian(L=L, R_I=rng.uniform(low=-L/2,high=L/2,size=(2,N_i*L**2))):
    '''1min 27s ± 3.7 s per loop (mean ± std. dev. of 7 runs, 1 loop each)'''
    _, _, kx, ky = get_k_space(L)

    sdotk = np.kron(sx, kx)+np.kron(sy, ky)
    H0 = vf * sdotk
    V_q = ft_potential_builder_3(L, R_I)
    return H0 + V_q


def conductivity_for_n(E, n, L, eta_factor=eta_factor):
    eta = eta_factor * vf * 2 * np.pi / L
    _, _, kx, ky = get_k_space(L)
    sxx = np.kron(np.eye(kx.shape[0]), sx)
    fd_diff = fermi_dirac(E[0]) - fermi_dirac(E[1])
    diff = E[0] - E[1]
    if diff == 0:
        return 0
    res = fd_diff / diff * ( n[0].T.conj() @ (sxx @ n[1]) ) * ( n[1].T.conj() @ (sxx @ n[0]) ) / ( diff + 1j * eta )
    # print(res)
    return res

def conductivity(L=L, eta_factor=eta_factor, R_I=rng.uniform(low=-L/2,high=L/2,size=(2,N_i*L**2))): # possibly the slowest function
    factor = -1j * 2 * np.pi * h_cut**2/L**2 * vf**2
    g_singular = 0
    eta = eta_factor * vf * 2 * np.pi / L
    ham = hamiltonian(L, R_I)
    if L == l_min:
        assert np.allclose(ham.T.conj(), ham)
        # assert 
    vals, vecs = np.linalg.eigh(ham) 
    '''np.linalg.eigh >>> 6.77 s ± 43.1 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)'''
    # for j in range(len(vals)):
    #     for k in range(-interaction_distance,interaction_distance):
    #         try:
    #             E = [vals[j], vals[j+k]]
    #             n = [vecs[j], vecs[j+k]]
    #             g_singular += conductivity_for_n(E, n, L, eta)
    #         except IndexError:
    #             pass
    for j in range(len(vals)):
        for k in range(len(vals)-j):
            E = [vals[j], vals[k]]
            n = [vecs[j], vecs[k]]
            g_singular += conductivity_for_n(E, n, L, eta)
    return g_singular * factor

def conductivity_vectorised(L=L, eta_factor=eta_factor, R_I=rng.uniform(low=-L/2,high=L/2,size=(2,N_i*L**2))): # possibly the slowest function
    '''
    999.97s on default function call: k_space_size = 45
    
    31.5 s ± 364 ms per loop (mean ± std. dev. of 7 runs, 1 loop each): k_space_size = 20
    '''
    factor = -1j * 2 * np.pi * h_cut**2/L**2 * vf**2

    eta = eta_factor * vf * 2 * np.pi / L

    ham = hamiltonian(L, R_I)
    if L == l_min:
        assert np.allclose(ham.T.conj(), ham)
        # assert 
    vals, vecs = np.linalg.eigh(ham) 
    '''np.linalg.eigh >>> 6.77 s ± 43.1 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)'''

    _, _, kx, ky = get_k_space(L)
    sxx = np.kron(np.eye(kx.shape[0]), sx)

    sx_vecs = sxx @ vecs # i checked this, matrix multiplication is equivalent to multiplying vector-wise 
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

    conductivities = np.array([conductivity_vectorised(l, eta_factor, rng.uniform(low=-l/2,high=l/2,size=(2,N_i*int(l)**2))) for l in L])

    print('conductivities computed')

    conductivities = str(conductivities.tolist())

    dirname = os.path.join('output_data','results_version','run'+sys.argv[1])
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

if __name__ == "__main__":

    # import mpi4py; rank = mpi4py.MPI.Comm().Get_rank()

    os.mkdir(os.path.join('output_data','results_version','run'+sys.argv[1]))
    
    L = [np.linspace(l_min, l_max,num_lengths)] * configurations

    with ProcessPoolExecutor() as exe:
        _ = exe.map(main, L)

    dirname = os.path.join('output_data','results_version','run'+sys.argv[1])
    
    fname = os.path.join('output_data','results_version','length.npy')

    if not os.path.isfile(fname):
        np.save(fname, L[0])
    if not os.path.isfile(os.path.join('output_data','results_version','params.txt')):
        with open(os.path.join('output_data','results_version','params.txt'),'w') as file:
            text = f'''l_min, l_max = {l_min}, {l_max}\neta = {eta_factor} * vf * 2 * pi / L\nvf = {vf}\nh_cut = {h_cut}\nu = {u}\nl0 = {l0}\nN_i = {N_i}\nT = {T}\nef = {ef}\nconfigurations = {configurations}\nk_space_size = {k_space_size}\nscattering potential = {function}'''#\na = {a}'''
            file.write(text)
            print('parameter file written')
    
   
if __name__=="__main__1":

    t0 = time.perf_counter()
    potential3 = ft_potential_builder_3()
    t2 = time.perf_counter()

    print('wrong block')
    assert 0
    
    print(f'{np.round(t2-t0,5)}s')



    plt.pcolormesh(np.flip(np.abs(potential3),0))#, cmap='RdBu_r')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(shrink=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join('graphics','full_hamiltonian.png'))
    # plt.show()

if __name__=="__main__1":
    # k1x, k1y, k2x, k2y = get_k_space(10)
    # k = np.kron(sy,k2y)
    # print(np.allclose(k, k.conj().T))
    ham = hamiltonian()
    assert np.allclose(ham, ham.T.conj())
    plt.subplot(1,2,1)
    # plt.pcolormesh(np.kron(sy, ky).imag)
    plt.pcolormesh(ham.imag)
    plt.subplot(1,2,2)
    # plt.pcolormesh(k.imag.T)
    plt.pcolormesh(ham.real)
    # plt.pcolormesh(np.kron(sy, ky).imag.T)
    plt.show()
