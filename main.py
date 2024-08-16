import numpy as np
from scipy import stats
from scipy.fft import fft2, rfft2
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import os, warnings, sys, time

l_min, l_max = 1e1,1e6

vf = 1 # 1e6
h_cut = 1
u = 1
l0 = l_min / 30
N_i = 20
L = 1e5
# l0 = L/30
eta = 1e6
T = 0
ef = 0
# a = 1

configurations = 50
interaction_distance = 3
k_space_size = 51
kernel_size = k_space_size
kernel_spread = 3
# eta = 1e5 * vf * 2 * np.pi / L

rng = np.random.default_rng()

sx = np.array([[0,1],[1,0]])
sy = 1j * np.array([[0,-1],[1,0]])

def gaussian_corr(q):
    return u * np.exp(-q**2*l0**2/2)

def thomas_fermi(q):
    return u / (q + l0**-1)

try:
    sys.argv[2]
    function = thomas_fermi
except IndexError:
    function = gaussian_corr
    
def get_k_space(L=L):
    '''
    k_space_size = 51
    161 μs ± 5.12 μs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
    '''
    lamda = 20*2*np.pi/L
    k_vec = np.linspace(-lamda, lamda, int(k_space_size**.5))
    
    cartesian_product = np.array(np.meshgrid(k_vec, k_vec, indexing='ij')).T.reshape(-1, 2)
    
    k1x, k2x = np.meshgrid(cartesian_product[:,0], cartesian_product[:,0])
    k1y, k2y = np.meshgrid(cartesian_product[:,1], cartesian_product[:,1])

    kx = k1x - k2x
    ky = k1y - k2y
    return kx, ky

def ft_potential_builder_3(L=L, R_I=rng.uniform(low=-L/2,high=L/2,size=(2,N_i))):

    '''
    k_space_size = 51
    >>> 4.14 ms ± 327 μs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    '''

    kx, ky = get_k_space(L)
    
    k_matrix = np.zeros_like(kx, dtype=np.complex128)
    
    for i in range(N_i):
        rands1 = np.ones_like(kx)*R_I[0,i]
        rands2 = np.ones_like(ky)*R_I[1,i]
        k_matrix += np.exp(1j * (kx * rands1 + ky * rands2)) * function( (kx**2 + ky**2)**.5 )

    return np.kron(np.eye(2),k_matrix)

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

def hamiltonian(L=L):
    '''
    4.4 ms ± 382 μs per loop (mean ± std. dev. of 7 runs, 100 loops each) -> k_space_size = 51
    '''
    kx, ky = get_k_space(L)
    H0 = vf * (np.kron(sx, kx) + np.kron(sy, ky))
    V_q = ft_potential_builder_3(L)
    return H0 + V_q


def conductivity_for_n(E, n, L, eta=eta):
    '''
    131 ms ± 2.01 ms per loop (mean ± std. dev. of 7 runs, 10 loops each) -> k_space_size = 2000
    '''
    eta = vf * 2 * np.pi / L
    kx, ky = get_k_space(L)
    sxx = np.kron(sx, np.eye(kx.shape[0]))
    fd_diff = fermi_dirac(E[0]) - fermi_dirac(E[1])
    diff = E[0] - E[1]
    if diff == 0:
        return 0
    res = fd_diff / diff * ( n[0].T.conj() @ (sxx @ n[1]) ) * ( n[1].T.conj() @ (sxx @ n[0]) ) / ( diff + 1j * eta )
    # print(res)
    return res

def conductivity(L=L, eta=eta, R_I=rng.uniform(low=-L/2,high=L/2,size=(2,N_i))): # possibly the slowest function
    '''
    1.95 s ± 440 ms per loop (mean ± std. dev. of 7 runs, 1 loop each) -> k_space_size = 51
    '''
    potential = ft_potential_builder_3(L, R_I)
    factor = -1j * 2 * np.pi * h_cut**2/L**2 * vf**2
    g_singular = 0
    ham = hamiltonian(L)
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

def main(L=np.linspace(l_min,l_max,15)): # faster locally (single node)

    conductivities = np.array([conductivity(l, eta, rng.uniform(low=-l/2,high=l/2,size=(2,N_i))) for l in L])

    conductivities = str(conductivities.real.tolist())

    dirname = os.path.join('output_data','run'+sys.argv[1])
    fname = determine_next_filename(fname='output',folder=dirname, filetype='txt')
    write_file(fname, conductivities)
    print('conductivities computed and stored')

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

    
    L = [np.linspace(l_min, l_max,3*5)] * configurations

    _ = [main(i) for i in L]
    
    dirname = os.path.join('output_data','run'+sys.argv[1])
    
    fname = determine_next_filename(fname='length',folder=dirname, filetype='npy')
    np.save(fname, L[0])
    if not os.path.isfile(os.path.join('output_data','params.txt')):
        with open(os.path.join('output_data','params.txt'),'w') as file:
            text = f'''l_min, l_max = {l_min}, {l_max}\nvf = {vf}\nh_cut = {h_cut}\nu = {u}\nl0 = {l0}\nN_i = {N_i}\neta = {eta}\nT = {T}\nef = {ef}\nconfigurations = {configurations}\nk_space_size = {k_space_size}\npotential = {function}'''#\na = {a}'''
            file.write(text)
            print('parameter file written')
    
   
if __name__=="__main__1":

    t0 = time.perf_counter()
    potential3 = ft_potential_builder_3()
    t2 = time.perf_counter()
    
    print(f'{np.round(t2-t0,5)}s')

    plt.pcolormesh(np.flip(np.abs(potential3),0))#, cmap='RdBu_r')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(shrink=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join('graphics','full_hamiltonian.png'))
    # plt.show()
