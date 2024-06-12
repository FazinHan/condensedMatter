import numpy as np
import time
import os
from concurrent.futures import ProcessPoolExecutor

k_space_size = 2000
Λ = lambda L: 20*2*np.pi/L
η = lambda L: 1e7*vf*2*np.pi/L # needs some determining

vf = 1
sx = np.array([[0,1],[1,0]])
sy = 1j * np.array([[0,-1],[1,0]])
h_cut = 1
Ef = 0

def fermi_dirac(x,T=1e7*vf*2*np.pi,ef=0):
    if T != 0:
        return 1/(1+np.exp((x-ef)/T))
    out = np.zeros(x.size)
    out[np.where(x < ef)] = 1
    out[np.where(x == ef)] = .5
    return out

# def fermi_dirac(E):
#     if Ef >= E:
#         return 1
#     elif Ef == E:
#         return .5
#     else:
#         return 0

def g_function(L):
    kx, ky = get_ks_points(L)
    terms = get_terms(kx, ky, Λ(L), η(L))
    out = 4*vf**2*np.pi*h_cut**2 / L**2 * np.sum( terms )
    return out

def g_function_parallel(eta):
    L = 10
    kx, ky = get_ks_points(L)
    terms = get_terms(kx, ky, Λ(L), eta, T = 0)
    # terms = get_terms(kx, ky, Λ(L), η(L))
    out = 4*vf**2*np.pi*h_cut**2 / L**2 * np.sum( terms )
    return out

def g_temperature_function_parallel(T):
    L = 10
    kx, ky = get_ks_points(L)
    terms = get_terms(kx, ky, Λ(L), η(L), T)
    # terms = get_terms(kx, ky, Λ(L), η(L))
    out = 4*vf**2*np.pi*h_cut**2 / L**2 * np.sum( terms )
    return out


def hamiltonian(kx,ky):
    return -1j * vf * (kx*sx + ky*sy)

def hamiltonian__(kx,ky):
    return -1j * vf * (kx*sx + ky*sy).T

def get_ks_points(L):
    kss = np.linspace(-20*2*np.pi/L,20*2*np.pi/L,k_space_size)
    kx, ky = np.meshgrid(kss,kss)
    return kx, ky#, num_points

def get_terms(kx,ky,Λ,eta,T=1e7*vf*2*np.pi):
    n = []
    n_ = []
    arr = []
    for k_x, k_y in zip(kx.ravel(),ky.ravel()):
        if k_x**2 + k_y**2 <= Λ**2:
            
            vals, vecs = np.linalg.eig(hamiltonian(k_x, k_y))
            vals_, vecs_ = np.linalg.eig(hamiltonian__(k_x, k_y))

            # enn_1 = vals.real[0]-vals_.real[0]
            # enn_2 = vals.real[1]-vals_.real[1]
            # nsn2_1 = np.abs(vecs[0].conj().T @ sx @ vecs_[0])**2
            # nsn2_2 = np.abs(vecs[1].conj().T @ sx @ vecs_[1])**2

            # to_append = 0

            # if enn_1 != 0:
            #     to_append += (fermi_dirac(vals[0])-fermi_dirac(vals_[0]))/enn_1 * nsn2_1 / (enn_1 + 1j*eta)
            # if enn_2 != 0:
            #     to_append += (fermi_dirac(vals[1])-fermi_dirac(vals_[1]))/enn_2 * nsn2_2 / (enn_2 + 1j*eta)
            
            for En, n in zip(vals, vecs):
                for En_, n_ in zip(vals_, vecs):
                    if abs(fermi_dirac(En)-fermi_dirac(En_)) > 1e-2:
                        enn_ = En - En_
                        nsn2 = np.abs(n.conj().T @ sx @ n_)**2
                        # print(fermi_dirac(En)-fermi_dirac(En_))
                        arr.append((fermi_dirac(En)-fermi_dirac(En_))/enn_ * nsn2 / (enn_ + 1j*eta))
            
    return np.array(arr)#, np.array(enn_)
        

if __name__ == '__main__':

    T = np.linspace(0,1e8)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(24) as exe:
        gs_func = exe.map(g_temperature_function_parallel,T)
    g_vs_T = np.array([i for i in gs_func])
    t1 = time.perf_counter()
    print('g vs T in {:.4f} seconds'.format(t1-t0))

    
    eta = np.linspace(50,η(10))
    t0 = time.perf_counter()
    with ProcessPoolExecutor(24) as exe:
        gs_func = exe.map(g_function_parallel,eta)
    g_vs_eta = np.array([i for i in gs_func])
    t1 = time.perf_counter()
    print('g vs η in {:.4f} seconds'.format(t1-t0))
    
    L = np.linspace(1e4,1e9)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(24) as exe:
        gs_func = exe.map(g_function,L)
    g_vs_L = np.array([i for i in gs_func])
    t1 = time.perf_counter()
    print('g vs L in {:.4f} seconds'.format(t1-t0))
    
    num = 1
    
    while os.path.isfile(f'./output data/g_vs_eta_L{num}.npz'):
        num += 1
    
    with open(f'./output data/g_vs_eta_L{num}.npz','wb') as file:
        np.savez(file, L=L, eta=eta, T=T, g_vs_eta=g_vs_eta, g_vs_L=g_vs_L, g_vs_T=g_vs_T)
