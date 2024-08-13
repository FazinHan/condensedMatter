import numpy as np
from scipy import stats
from scipy.fft import fft2, rfft2
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import os, warnings, sys, time

l_min, l_max = 0.01,1e3

vf = 1 # 1e6
h_cut = 1
u = 1
l0 = l_min / 30
N_i = 10
L = 1e5
# l0 = L/30
eta = 1e6
T = 0
ef = 0
a = 1

configurations = 50
interaction_distance = 3
k_space_size = 51
# k_space_size = 20
kernel_size = k_space_size
kernel_spread = 3
# eta = 1e5 * vf * 2 * np.pi / L

rng = np.random.default_rng()

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
    k_space_size = 51
    161 μs ± 5.12 μs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
    '''
    lamda = 20*2*np.pi/L
    k_vec = np.linspace(-lamda, lamda, int(k_space_size**.5))
    
    cartesian_product = np.array(np.meshgrid(k_vec, k_vec, indexing='ij')).T.reshape(-1, 2)
    # cartesian_product = cartesian_product[np.where(cartesian_product[:,0]**2+cartesian_product[:,1]**2 <= lamda**2)]
    
    k1x, k2x = np.meshgrid(cartesian_product[:,0], cartesian_product[:,0])
    k1y, k2y = np.meshgrid(cartesian_product[:,1], cartesian_product[:,1])

    kx = k1x - k2x
    ky = k1y - k2y
    return kx, ky

def ft_potential_builder_3(L=L, R_I=rng.uniform(high=a,size=(2,N_i))):

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
        # plt.matshow(np.abs(k_matrix))
        # plt.show()

    return np.kron(np.eye(2),k_matrix)

def fermi_dirac_ondist(x,T=T,ef=ef): # T=1e7*vf*2*np.pi
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

def conductivity(L=L, eta=eta, R_I): # possibly the slowest function
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

def main2(L=L):

    cond = 0
    # conductivities = np.array([conductivity(l, eta) for l in L])
    for i in range(configurations):
        cond += conductivity(L, eta)
    
    return cond / configurations

def main(L=np.linspace(l_min,l_max,15)): # faster locally (single node)

    # cond = 0
    R_I = rng.uniform(high=a,size=(2,N_i))
    conductivities = np.array([conductivity(l, eta, R_I) for l in L])
    # for i in range(configurations):
        # cond += conductivity(L, eta)

    # assert np.alltrue(conductivities.imag < 1e-21)

    conductivities = str(conductivities.real.tolist())

    # return cond / configurations
    dirname = os.path.join('output_data','run'+sys.argv[1])
    fname = determine_next_filename(fname='output',folder=dirname, filetype='txt')
    write_file(fname, conductivities)
    print('conductivities computed and stored')
    # return conductivities

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
    if not os.path.isfile(os.path.join('output_data','params','params.txt')):
        os.mkdir(os.path.join('output_data','params'))
        with open(os.path.join('output_data','params','params.txt'),'w') as file:
            text = f'''l_min, l_max = {l_min}, {l_max}\nvf = {vf}\nh_cut = {h_cut}\nu = {u}\nl0 = {l0}\nN_i = {N_i}\neta = {eta}\nT = {T}\nef = {ef}\nconfigurations = {configurations}\nk_space_size = {k_space_size}\na = {a}'''
            file.write(text)
            print('parameter file written')
    
    # print(L)
    # ones = np.ones(configurations)
    # l, _ = np.meshgrid(L, ones)

    # try:
    #     for i in L:
    #         os.remove(os.path.join('')

    # t0 = time.perf_counter()

    # with ProcessPoolExecutor(40) as exe:
        # conductivities = list(exe.map(main, l))


    # with ProcessPoolExecutor() as exe:
    #     conductivities = list(exe.map(main,L))

    # print(conductivities.shape)
    # cs = CubicSpline(L, conductivities)
    # t1 = time.perf_counter()
    # print('%.2f s'%(t1-t0))
    # try:
    #     name = sys.argv[1]
    # except IndexError:
    #     name = 'data'

    # name = determine_next_filename(name,'npz','output data')
    # with open(name,'wb') as file:
    #     np.savez(file, L=L, eta=eta, conductivities=conductivities, beta=cs(L,1))
    #     print('data written to',name)

    # try:
    #     name = sys.argv[2]
    # except IndexError:
    #     name = 'output'

    # import plotter; plotter.main(name)

    
    # np.save(fname, conductivities)
    # print(np.array(conductivities))
    # print(f"data written to {fname}")

    # plotter(L, conductivities, beta)
    # g = cs(L)

if __name__=="__main__1":
    # potential3 = hamiltonian()

    t0 = time.perf_counter()
    potential3 = ft_potential_builder_3()
    # t1 = time.perf_counter()
    # potential4 = conductivity()
    t2 = time.perf_counter()
    
    # print(potential3)
    # print(potential4)
    # print(f'{np.round(t1-t0,5)}s using jit')
    print(f'{np.round(t2-t0,5)}s')

    # fig, axs = plt.subplots(2,1,sharex=True)
    
    # axs[0].matshow(potential3.real, cmap='RdBu_r')
    # # axs[0,1].matshow(np.abs(potential3), cmap='RdBu_r')
    # pcm = axs[1].matshow(potential3.imag, cmap='RdBu_r')
    # axs[0].set_ylabel('real part')
    # axs[1].set_ylabel('imaginary part')
    # for ax in axs:   
    #     ax.axes.get_xaxis().set_ticks([])
    #     ax.axes.get_yaxis().set_ticks([])
    # fig.colorbar(pcm, ax=axs, shrink=0.6)
    # plt.colorbar(loc='bottom')
    # print(np.allclose(potential3.real, np.abs(potential3)))

    plt.pcolormesh(np.flip(np.abs(potential3),0))#, cmap='RdBu_r')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(shrink=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join('graphics','full_hamiltonian.png'))
    # plt.show()
