#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 13:08:11 2024

@author: swadeepannanda
"""

import numpy as np
import random
L = 20 #system size
K = 20*(2*np.pi/L)#momentum cut off
k_grid = 50 # number of kpoints
n_config = 50 #number of potential centers
l0 = 1
u0 = 1
eta = 1.E-6
epsilon = 1.E-6
h2 = np.zeros((k_grid*k_grid, k_grid*k_grid), dtype = complex,)
h1x = np.zeros((k_grid*k_grid, k_grid*k_grid), dtype = complex,)
h1y = np.zeros((k_grid*k_grid, k_grid*k_grid), dtype = complex,)
h3 = np.zeros((k_grid*k_grid, k_grid*k_grid), dtype = complex,)
k_disc = []
#centers = [random.uniform(-20, 20) for _ in range(100)]
strength = []



kx = np.linspace(-K, K, k_grid)
ky = np.linspace(-K, K, k_grid)
sigmax = np.array([[0,1],[1,0]])
sigmay = np.array([[0, -1j], [1j, 0]])
sigmaz = np.array([[1, 0], [0, -1]])
sigma0 = np.eye(2)

def generate_random_tuple(min_val, max_val):
    return (random.uniform(min_val, max_val), random.uniform(min_val, max_val))


def h0(kx1, ky1):
    return kx1* sigmax + ky1*Ik, sigmay

def hv(k1, k2, R, u0, l0): #To generate the potential matrix. k1, k2 are the bra and ket momentum states, R is the impurity location 
    kx = (k1[0] - k2[0])
    ky = (k1[1] - k2[1])
    x = u0*np.exp(1j*(kx*R[0] + 1j*ky*R[1]))*np.exp(-(kx**2 + ky**2)*((l0)**2)/2)
    y = (1j*R[0]-l0**2*kx)*x
    return x, y






def htheta(x):
    return 1 if x >=0 else 0
def g(L):
    centers = [generate_random_tuple(-K, K) for _ in range(n_config)]
    h4 = np.zeros((2*k_grid*k_grid, 2*k_grid*k_grid))
    h5 = np.zeros((2*k_grid*k_grid, 2*k_grid*k_grid))
    for i in range(len(k_disc)):
        for j in range(len(k_disc)):
    #        h2[i][j] = hv(i, j, R, u0, l0)[0]
    #        h3[i][j] = hv(i, j, R, u0, l0)[1]
            h1x[i][i] = k_disc[i][0]
            h1y[j][j] = k_disc[i][1]
            

    for R in centers:
        for i in range(len(k_disc)):
            for j in range(len(k_disc)):
                h2[i][j] = hv(k_disc[i], k_disc[j], R, u0, l0)[0]
                h3[i][j] = hv(k_disc[i], k_disc[j], R, u0, l0)[1]
             #   h1[i][j] = i
        h4 = h4 + np.kron(h2, sigma0)
        h5 = h5 + np.kron(h3, sigma0)
      #  h1 = np.kron(h1, sigmax) + np.kron(h1, sigmay)   
    h =   np.kron(h1x, sigmax) + np.kron(h1y, sigmay) + h4
    dh = np.kron(Ik, sigmax) + np.kron(Ik, sigmay) + h5

    eigenval, eigvec = np.linalg.eigh(h)
    tem = 0
    for i in range(2*k_grid*k_grid):
        for j in range(2*k_grid*k_grid):
            vec_i = eigvec[i]
            vec_j = eigvec[j]
            x = (htheta(eigenval[i]) - htheta(eigenval[j]))/(eigenval[i] - eigenval[j] + epsilon)
            y = np.dot(np.conjugate(vec_i).T, np.dot(dh, vec_j))*np.dot(np.conjugate(vec_j).T, np.dot(dh, vec_j))/(eigenval[i] - eigenval[j] + 1.j*eta)
            tem += x*y
    return tem

if __name__=="__main__":
    for i in kx:
        for j in ky:
            k_disc.append((i,j))
            
    
    Ik = np.eye(k_grid*k_grid)
        

        
        
       

#def beta(f, x):
##   return (f(x+h) - f(x-h))/(2*h)        

    print(g(20))                
    
    
    




        