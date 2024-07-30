import numpy as np
from scipy import stats
from scipy.fft import fft2, rfft2
from scipy.interpolate import CubicSpline
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import os, warnings, sys, time

l_min, l_max = 1, 1e10

vf = 1 # 1e6
h_cut = 1
u = 1
l0 = l_min / 30
N_i = 10
L = l_min
# l0 = L/30
eta = 1e-4

configurations = 20
k_space_size = 2000
# k_space_size = 20
kernel_size = k_space_size
kernel_spread = 3
# eta = 1e5 * vf * 2 * np.pi / L

rng = np.random.default_rng(128)

sx = np.array([[0,1],[1,0]])
sy = 1j * np.array([[0,-1],[1,0]])


def gaussian_corr_inverse(x):
    return np.sqrt(2/l0**2*np.log(u/x))

def thomas_fermi_inverse(x):
    return u/x-1/l0

def gaussian_corr(q):
    return u * np.exp(-q**2*l0**2/2)

def thomas_fermi(q):
    return u / (q + l0**-1)

def kernel_builder(size, function):
    axis = np.linspace(-kernel_spread, kernel_spread, size)
    x, y = np.meshgrid(axis, axis)
    kernel = np.zeros_like(x)
    with np.nditer(kernel, flags=['multi_index']) as it:
        for _ in it:
            kernel[it.multi_index] += function(x[it.multi_index],y[it.multi_index])
    return kernel

try:
    sys.argv[1]
    randomness = thomas_fermi_inverse
    function = thomas_fermi
except IndexError:
    randomness = gaussian_corr_inverse
    function = gaussian_corr
    
def ft_potential_builder(L, N_i, k_space_size, kernel_size, function):
    r = np.random.uniform(size=(k_space_size,k_space_size))
    r = function(r)
    return fft2(r)

def ft_potential_builder_2(L, N_i, k_space_size, kernel_size, function):
    kernel = kernel_builder(kernel_size, function)
    rng = np.random.Generator(np.random.PCG64())
    indices = rng.integers(0, k_space_size, size=(N_i, 2))
    r = np.zeros((k_space_size, k_space_size))
    # r[indices[:,0],indices[:,1]] = 1
    r[indices[:,0],indices[:,1]] = 1
    
    # print(indices)
    result = fft2(r) * kernel

    # print(result.shape)
    
    return result

def ft_potential_builder_2_5(L=L):

    lamda = 20*2*np.pi/L
    k_vec = np.linspace(-lamda, lamda, k_space_size)
    
    k_diag = np.diag(k_vec)
    kxx, kyy = np.meshgrid(k_vec, k_vec)
    # k2 = (kx**2 + ky**2)**.5
    k_matrix = np.zeros_like(np.kron(kxx,kyy), dtype=np.complex128)
    # return kx - ky

    for kx in range(k_matrix.shape[0]):
        for ky in range(k_matrix.shape[1]):
            for i in range(N_i):
                rnd_num1 = rng.standard_normal()
                rnd_num2 = rng.standard_normal()
                k_matrix[kx,ky] += np.exp(1j * (kxx[kx,ky] - kyy[kx,ky]) * rnd_num) * function( (kxx[kx,ky]**2 + kyy[kx,ky]**2)**.5 )
                # plt.matshow(np.abs(k_matrix))
                # plt.show()

    return np.kron(np.eye(2),k_matrix)

def get_k_space(L=L):
    '''
    k_space_size = 2000
    58.7 ms ± 906 μs per loop (mean ± std. dev. of 7 runs, 10 loops each)
    '''
    lamda = 20*2*np.pi/L
    k_vec = np.linspace(-lamda, lamda, int(k_space_size**.5))
    
    cartesian_product = np.array(np.meshgrid(k_vec, k_vec, indexing='ij')).T.reshape(-1, 2)
    cartesian_product = cartesian_product[np.where(cartesian_product[:,0]**2+cartesian_product[:,1]**2 <= lamda**2)]
    
    k1x, k2x = np.meshgrid(cartesian_product[:,0], cartesian_product[:,0])
    k1y, k2y = np.meshgrid(cartesian_product[:,1], cartesian_product[:,1])

    kx = k1x - k2x
    ky = k1y - k2y
    return kx, ky

def ft_potential_builder_3(L=L):

    '''
    k_space_size = 2000
    >>> 7.81 s ± 36.2 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    matrices shape (1436, 1436) after truncation of space
    '''

    kx, ky = get_k_space(L)
    
    k_matrix = np.zeros_like(kx, dtype=np.complex128)
    
    for i in range(N_i):
        rands1 = rng.standard_normal(kx.shape)
        rands2 = rng.standard_normal(ky.shape)
        k_matrix += np.exp(1j * (kx * rands1 + ky * rands2)) * function( (kx**2 + ky**2)**.5 )
        # plt.matshow(np.abs(k_matrix))
        # plt.show()

    return np.kron(np.eye(2),k_matrix)

def fermi_dirac_ondist(x,T=0,ef=0): # T=1e7*vf*2*np.pi
    if T != 0:
        return 1/(1+np.exp((x-ef)/T))
    out = np.zeros(x.shape)
    out[np.where(x < ef)] = 1
    out[np.where(x == ef)] = .5
    return out

def fermi_dirac(x,T=0,ef=0):
    if T != 0:
        return 1/(1+np.exp((x-ef)/T))
    if x>ef:
        return 0
    if x==ef:
        return .5
    return 1

def hamiltonian(L):
    kx, ky = get_k_space(L)
    # lamda = 20*2*np.pi/L
    # k_vec = np.linspace(-lamda, lamda, k_space_size)
    # k_diag = np.diag(k_vec)
    H0 = vf * np.kron(sx+sy, k_diag)
    V_q = ft_potential_builder_3(L)
    return H0 + V_q

def conductivity_for_n(E, n, L, eta):
    sxx = np.kron(sx, np.eye(k_space_size**2))
    fd_diff = fermi_dirac(E[0]) - fermi_dirac(E[1])
    operator = sxx @ n[1] @ n[1].T.conj() @ sxx
    density = n[0] @ n[0].T.conj()
    diff = E[0] - E[1]
    if diff == 0:
        return 0
    res = fd_diff / diff * np.trace( density @ operator ) / ( diff + 1j * eta )
    return res

def conductivity(k_space_size, L, eta, function):
    # lamda = 20*2*np.pi/L
    # k = np.linspace( -lamda, lamda, int(k_space_size**.5) )
    potential = ft_potential_builder_3(L)
    factor = -1j * 2 * np.pi * h_cut**2/L**2 * vf**2
    g_singular = 0
    for i in range(configurations):
        ham = hamiltonian(L)
        vals, vecs = np.linalg.eigh(ham)
        for j in range(len(vals)):
            for k in range(len(vals)-j):
                E = [vals[j], vals[k]]
                n = [vecs[j], vecs[k]]
                g_singular += conductivity_for_n(E, n, L, eta)
    return g_singular * factor

def main(L):
        
    conductivities_summed = 0
    for i in range(configurations):
        conductivities_summed += conductivity(k_space_size, L, eta, function)
    if conductivities_summed.imag >= 1e-4:
        warnings.warn(f'conductivity unreal {conductivities_summed}')
    return conductivities_summed / configurations

def determine_next_filename(fname='output',filetype='png',folder='graphics',exists=False):
    num = 1
    filename = lambda num: f'{fname}{num}.{filetype}'
    while os.path.isfile(os.path.join('.',folder,filename(num))):
        num += 1
    else:
        if exists:
            num -= 1
    return os.path.join('.',folder,filename(num))

def plotter(L, conductivities, beta, save, name='output'):
    fig, axs = plt.subplots(2,1)
    # t2 = time.perf_counter()
    # print(f'parallelised: {np.round(t2-t1,3)}s taken')
    
    axs[1].plot(L, conductivities,'.')
    axs[0].plot(conductivities, beta,'.')
    # axs[0].set_xlabel('$\\eta$')
    axs[1].set_xlabel('L')
    axs[1].set_ylabel('g')
    axs[0].set_xlabel('g')
    axs[0].set_ylabel('$\\beta$')
    if name=='output':    
        fig.suptitle(randomness.__name__[:-8])
    else:
        fig.suptitle(name)
    fig.tight_layout()
    name = determine_next_filename(name)#fname='eta_variance')
    if save:
        plt.savefig(name)
        print('plotted to',name)
    else:
        plt.show()

if __name__ == "__main__1":

    
    L = np.linspace(l_min, l_max,3*5)
    conductivities = []
    
    # import time

    t0 = time.perf_counter()

    # cs = CubicSpline(L, [main(i) for i in L])

    # print(f'serialised: {np.round(t1-t0,3)}s taken')

    with ProcessPoolExecutor(14) as exe:
        conductivities = [i for i in exe.map(main, L)]
    cs = CubicSpline(L, conductivities)
    t1 = time.perf_counter()
    print('%.2f s'%(t1-t0))
    try:
        name = sys.argv[1]
    except IndexError:
        name = 'data'

    name = determine_next_filename(name,'npz','output data')
    with open(name,'wb') as file:
        np.savez(file, L=L, eta=eta, conductivities=conductivities, beta=cs(L,1))
        print('data written to',name)

    try:
        name = sys.argv[1]
    except IndexError:
        name = 'output'

    import plotter; plotter.main(name)

    # plotter(L, conductivities, beta)
    # g = cs(L)

if __name__=="__main__":
    potential3 = ft_potential_builder_3()
    
    print(potential3.shape)
    
    plt.matshow(np.abs(potential3))
    plt.colorbar()
    plt.show()