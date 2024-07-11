import numpy as np
from scipy import stats
from scipy.fft import fft2, rfft2
from scipy.interpolate import CubicSpline
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import os, warnings, sys

'''right now the potential is not fourier transformed: try to see what happens when transformed'''

# serialised: 36.035s taken
# parallelised: 4.797s taken

vf = 1
h_cut = 1
u = 1
l0 = 1
N_i = 10
L = 1e-8
eta = 1e5

configurations = 50
k_space_size = 100
# k_space_size = 20
kernel_size = k_space_size
kernel_spread = 3
# eta = 1e5 * vf * 2 * np.pi / L


sx = np.array([[0,1],[1,0]])
sy = 1j * np.array([[0,-1],[1,0]])


def gaussian_corr_inverse(x):
    return np.sqrt(2/l0**2*np.log(u/x))

def thomas_fermi_inverse(x):
    return u/x-1/l0

def gaussian_corr(x, y):
    return u * np.exp(-(x**2+y**2)*l0**2/2)

def thomas_fermi(x, y):
    return u / ((x**2+y**2)**.5 + l0**-1)

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
    
def ft_potential_builder(N_i, k_space_size, kernel_size, function):
    r = np.random.uniform(size=(k_space_size,k_space_size))
    r = function(r)
    return fft2(r)

def ft_potential_builder_2(N_i, k_space_size, kernel_size, function):
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

def hamiltonian(kx,ky,potential):
    return vf * (kx*sx + ky*sy) + np.eye(2)*potential

def eig_n(hamiltonian):
    vals, vecs = np.linalg.eigh(hamiltonian)
    return vals, vecs

def conductivity_for_n(E, n, L, eta):
    factor = -1j * 2 * np.pi * h_cut**2/L**2 * vf**2
    fd_diff = fermi_dirac(E[0]) - fermi_dirac(E[1])
    operator = sx @ n[1] @ n[1].T.conj() @ sx
    density = n[0] @ n[0].T.conj()
    diff = E[0] - E[1]
    if diff == 0:
        return 0
    res = factor * fd_diff / diff * np.trace( density @ operator ) / ( diff + 1j * eta )
    return res

def conductivity(k_space_size, L, eta, function):
    lamda = 20*2*np.pi/L
    k = np.linspace( -lamda, lamda, k_space_size )
    kxx, kyy = np.meshgrid(k, k)
    # potential = ft_potential_builder(k_space_size, function)
    potential = ft_potential_builder_2(N_i, k_space_size, kernel_size, function)
    # plt.matshow(potential)
    # plt.show()
    g_singular = 0
    for kx in range(kxx.shape[0]):
        for ky in range(kyy.shape[0]):
            if kxx[kx, ky]**2+kyy[kx, ky]**2 < lamda**2:
                ham = hamiltonian(kxx[kx, ky], kyy[kx, ky], potential[kx, ky])
                E, n = np.linalg.eig(ham)
                
                assert np.allclose(E[0], potential[kx,ky]+(kxx[kx, ky]**2+kyy[kx, ky]**2)**.5) or np.allclose(E[0], potential[kx,ky]-(kxx[kx, ky]**2+kyy[kx, ky]**2)**.5), f'\n\neigenvalue miscompute: {E[0]} != {potential[kx,ky]} +- {(kxx[kx, ky]**2+kyy[kx, ky]**2)**.5}' 
                
                n = [i.reshape((2,1)) for i in n]
                g_singular += conductivity_for_n(E, n, L, eta)
        # print(f'kxx layer {kx} complete')
    return g_singular

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
    
    axs[0].loglog(L, conductivities,'.')
    axs[1].plot(conductivities, beta,'.')
    # axs[0].set_xlabel('$\\eta$')
    axs[0].set_xlabel('L')
    axs[0].set_ylabel('g')
    axs[1].set_xlabel('g')
    axs[1].set_ylabel('$\\beta$')
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

if __name__ == "__main__":

    
    L = np.logspace(-14,0,3*5)
    conductivities = []
    
    # import time

    # t0 = time.perf_counter()

    # cs = CubicSpline(L, [main(i) for i in L])

    # t1 = time.perf_counter()
    # print(f'serialised: {np.round(t1-t0,3)}s taken')

    with ProcessPoolExecutor(14) as exe:
        conductivities = [i for i in exe.map(main, L)]
        
    cs = CubicSpline(L, conductivities)

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

if __name__=="__main__1":
    potential = ft_potential_builder_2(N_i, k_space_size, kernel_size, gaussian_corr)
    print("potential built")
    plt.matshow(potential.real)
    plt.show()
