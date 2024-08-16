import numpy as np
from scipy import stats
from scipy.fft import fft2, rfft2
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import os, warnings, sys, time

eta_min, eta_max = 1e1,1e25

vf = 1 # 1e6
h_cut = 1
u = 1
N_i = 20
L = 1e2
l0 = L/30
T = 0
ef = 0
# a = 1

configurations = 1
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
    # V_q = ft_potential_builder_3(L)
    return H0 #+ V_q


def conductivity_for_n(E, n, L, eta):
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

def conductivity(eta, L=L):#, R_I=rng.uniform(low=-L/2,high=L/2,size=(2,N_i))): # possibly the slowest function
    '''
    1.95 s ± 440 ms per loop (mean ± std. dev. of 7 runs, 1 loop each) -> k_space_size = 51
    '''
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

def main_eta(eta=np.linspace(eta_min,eta_max,15)): # faster locally (single node)

    conductivities = np.array([conductivity(e, L) for e in eta])

    return conductivities

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

def plotter(L, conductivities, beta, save, name='output', folder=''):
    fig, axs = plt.subplots(2,1)
    # t2 = time.perf_counter()
    # print(f'parallelised: {np.round(t2-t1,3)}s taken')
    
    axs[1].plot(L, conductivities,'.')
    axs[0].plot(conductivities, beta,'.')
    # axs[0].set_xlabel('$\\eta$')
    axs[1].set_xlabel('$\\eta$')
    axs[1].set_ylabel('g')
    axs[0].set_xlabel('g')
    axs[0].set_ylabel('$\\beta$')
    # if name=='output':    
    #     fig.suptitle(randomness.__name__[:-8])
    # else:
    #     fig.suptitle(name)
    fig.tight_layout()
    name = determine_next_filename('eta-plot', folder=folder, filetype='png')#fname='eta_variance')
    if save:
        plt.savefig(name)
    else:
        plt.show()
    return name

if __name__ == "__main__":
    
    
    eta = np.linspace(eta_min, eta_max,3*5)

    conductivities = main_eta(eta)
    
    dirname = 'output_data'
    
    cs = CubicSpline(eta, conductivities)
    g = cs(eta)
    beta = cs(eta, 1)
    name = plotter(eta, g, beta, 1, folder=dirname)

    fname = name.strip('.png')+'_params.txt'

    with open(fname,'w') as file:
        text = f'''eta_min, eta_max = {eta_min}, {eta_max}\nvf = {vf}\nh_cut = {h_cut}\nu = {u}\nl0 = {l0}\nN_i = {N_i}\nL = {L}\nT = {T}\nef = {ef}\nconfigurations = {configurations}\nk_space_size = {k_space_size}\npotential = {function}'''#\na = {a}'''
        file.write(text)
        print('parameter file written')