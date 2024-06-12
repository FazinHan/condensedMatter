import numpy as np
from numpy import exp
import os
import matplotlib.pyplot as plt
import time
from concurrent.futures import ProcessPoolExecutor

k_space_size = 2000
Lambda = lambda L: 20*2*np.pi/L
# eta = lambda L: 1e7*vf*2*np.pi/L # needs some determining

vf = 1
sx = np.array([[0,1],[1,0]])
sy = 1j * np.array([[0,-1],[1,0]])
h_cut = 1
Ef = 0
k_B = 1

u = 1e5
l0 = 1

def g_function(L):
    eta = 1e5*vf*2*np.pi/L
    T = 1e3*vf*2*np.pi/L
    lamda = Lambda(L)
    k = np.linspace(-lamda, lamda, k_space_size)
    kx, ky = np.meshgrid(k, k)
    # print(kx.shape[0], ky.shape[1])
    summ = 0
    for i in range(kx.shape[0]):
        for j in range(ky.shape[1]):
            k = (kx[i,j]**2 + ky[i,j]**2)**.5 
            if k**2 <= lamda**2:
                # print(k, lamda)
                summ += ky[i,j]**2 /k**(3/2) /(2*k+1j*eta) * ( 1/(np.exp(vf*k/T)+1) - 1/(np.exp(-vf*k/T)+1) )
    return -8j * np.pi * h_cut**2 / vf / L**2 * summ

def g_eta(eta):
    L = 1
    T = 1e3*vf*2*np.pi/L
    lamda = Lambda(L)
    k = np.linspace(-lamda, lamda, k_space_size)
    kx, ky = np.meshgrid(k, k)
    # print(kx.shape[0], ky.shape[1])
    summ = 0
    for i in range(kx.shape[0]):
        for j in range(ky.shape[1]):
            k = (kx[i,j]**2 + ky[i,j]**2)**.5 
            if k**2 <= lamda**2:
                # print(k, lamda)
                summ += ky[i,j]**2 /k**(3/2) /(2*k+1j*eta) * ( 1/(np.exp(vf*k/T)+1) - 1/(np.exp(-vf*k/T)+1) )
    return -8j * np.pi * h_cut**2 / vf / L**2 * summ


if __name__ == '__main__':
    L = np.linspace(1e4,1e9)

    t0 = time.perf_counter()
    with ProcessPoolExecutor(24) as exe:
        gs_func = exe.map(g_function, L)
    g_vs_L = np.array([i for i in gs_func])
    t1 = time.perf_counter()
    print('g vs L in {:.4f} seconds'.format(t1-t0))

    eta = np.linspace(0.01,1e4)

    t0 = time.perf_counter()
    with ProcessPoolExecutor(24) as exe:
        gs_func = exe.map(g_eta, eta)
    g_vs_eta = np.array([i for i in gs_func])
    t1 = time.perf_counter()
    print('g vs η in {:.4f} seconds'.format(t1-t0))
    
    
    # g = [g_function(i, T=1e3*vf*2*np.pi/i, eta=1e3*vf*2*np.pi/i) for i in L]
    
    num = 1
    
    while os.path.isfile(f'./output_data/g_vs_eta_L{num}.npz'):
        num += 1
    
    with open(f'./output_data/g_vs_eta_L{num}.npz','wb') as file:
        np.savez(file, L=L, g_vs_L=g_vs_L, eta=eta, g_vs_eta=g_vs_eta)

    print('complete')