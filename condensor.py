from main import determine_next_filename
import sys, os
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

def main():
    conductivities = []

    directory = os.path.join('output_data','results_version')

    lfname = os.path.join('output_data','results_version','length.npy')
    
    for root, _, fnames in os.walk(directory):
        for fname in fnames:
            with open(os.path.join(root, fname),'r') as file:
                data = eval(file.read())
                conductivities.append(data)
    conductivities = np.array(conductivities)
    conductivities = np.sum(conductivities, axis=0)/conductivities.shape[0]
    
    #lfname = os.path.join(directory,'length1.npy')
    with open(lfname,'rb') as file:
        L = np.load(file)

    fname = determine_next_filename(folder=os.path.join('output_data','data'),fname='L_cond_data',filetype='txt')

    with open(fname, 'w') as file:
        file.write(L)
        file.write(conductivities)

if __name__=="__main__":
    main()
    # import mpi4py; rank = mpi4py.MPI.Comm().Get_rank()
    
