import numpy as np
from qutip import *
# from numpy.linalg import eig
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.widgets import Slider

N = 100

X = np.linspace(-10, 10, N)
Y = np.linspace(-10, 10, N)
x, y = np.meshgrid(X, Y)
# z = np.sin(x) + np.sin(y)

# sx = np.array()

def ham(kx,ky,hz):
    sol = np.copy(kx)
    sol1 = np.copy(kx)
    for i,j in zip(range(kx.shape[0]),range(ky.shape[1])):
        hamiltonian = ky[i,j]*sigmax() - kx[i,j]*sigmay() + hz*sigmaz()
        sol[i,j] = hamiltonian.eigenenergies()[0]
        sol1[i,j] = hamiltonian.eigenenergies()[1]
    return sol, sol1


fig = plt.figure()

ax1 = fig.add_axes([0, 0, 1, 0.8], projection = '3d')
ax2 = fig.add_axes([0.1, 0.85, 0.8, 0.1])

s = Slider(ax = ax2, label = 'value', valmin = 0, valmax = 5, valinit = 2)

def update(val):
    value = s.val
    ax1.cla()
    surfaces = ham(x,y,value)
    ax1.plot_surface(x, y, surfaces[0], cmap = cm.coolwarm, linewidth = 0, antialiased = False)
    ax1.plot_surface(x, y, surfaces[1], cmap = cm.coolwarm, linewidth = 0, antialiased = False)
    ax1.set_zlim(-2, 7)

s.on_changed(update)
update(0)

plt.show()