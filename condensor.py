from main import determine_next_filename
import os
import numpy as np
from mpi4py import MPI

def main(path):
    conductivities = []

    directory = os.path.join('output_data','results_version',path)

    for root, _, fnames in os.walk(directory):
        for fname in fnames:
            with open(os.path.join(root, fname),'r') as file:
                data = eval(file.read())
                conductivities.append(data)

    return np.array(conductivities)

def parallelise():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  # Process ID (rank)

    path = 'run'+str(rank)
    # Each process computes its part
    result = main(path)

    # Gather all results to the root process (process 0)
    results = comm.gather(result, root=0)

    if rank == 0:
        lfname = os.path.join('output_data','results_version','length.npy')
        with open(lfname,'rb') as file:
            L = np.load(file)

        conductivities = np.mean(results, axis=0)

        sem = np.std(results, axis=0) / np.sqrt(len(results))

        fname = determine_next_filename(folder=os.path.join('output_data','data'),fname='L_cond_data',filetype='npz')

        with open(fname, 'wb') as file:
            np.savez(file,L=L, conductivities=conductivities, sem=sem)


if __name__=="__main__":
    parallelise()
