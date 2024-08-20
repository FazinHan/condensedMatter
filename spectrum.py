from main import hamiltonian, get_k_space
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

ham = hamiltonian()

_, _, kx, ky = get_k_space()
kx = np.diag(kx)
ky = np.diag(ky)
k = np.sqrt(kx**2+ky**2)

vals, vecs = np.linalg.eigh(ham)

# vals = np.unique(vals)

dataframe = pd.DataFrame(vals)

# print(dataframe)

plt.plot(k, vals[:2025],'.')
plt.plot(k, vals[2025:],'.')
plt.xlabel('k')
plt.ylabel('Eigenvalues')
plt.show()
# print(k.size)