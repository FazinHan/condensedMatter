import numpy as np
from mpi4py import MPI

rng = np.random.default_rng()

rand = rng.uniform(size=(10,10))

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    results = comm.gather(rand, root=0)

    print(results)

    assert not np.allclose(results[0], results[1])

    print('Randomiser is thread-safe\n')

