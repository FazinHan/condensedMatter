import numpy as np

l_min, l_max = 10,40
num_lengths = 15

vf = 1
h_cut = 1
u = 10
l0 = l_min / 30
N_i = 10
L = 10
# l0 = L/30
eta_factor = 100
T = 0
ef = 0

configurations = 1
k_space_size = 21

interaction_distance = 3

sx2 = np.array([[0,1],[1,0]])
sy2 = 1j * np.array([[0,-1],[1,0]])
sz2 = np.eye(2)
sz2[-1,-1] = -1
