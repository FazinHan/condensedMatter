from main import determine_next_filename, plotter
import sys, os
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

def main(name):
    conductivities = []
    dirname = 'output_data'
    version = determine_next_filename('results_version',folder='output_data',direc=True,exists=True)
    
    for root, _, fnames in os.walk(version):
    # for root, _, fnames in os.walk(dirname):
        for fname in fnames:
            with open(os.path.join(root, fname),'rb') as file:
                data = np.load(file)
                # print(1)
                conductivities.append(data)
    conductivities = np.array(conductivities)
    conductivities = np.sum(conductivities, axis=0)
    
    fname = os.path.join(version, 'run1','length1.npy')
    with open(fname,'rb') as file:
        L = np.load(file)

    # print(conductivities.shape)
    # print(L.shape)

    # print(L.shape)
    cs = CubicSpline(L, conductivities)
    g = cs(L)
    beta = cs(L, 1)
    plotter(L[:-1], np.abs(g)[:-1], np.abs(beta)[:-1], name!='output', name)
    # plotter(L, np.abs(g), np.abs(beta), name!='output', name)

if __name__=="__main__":
    try:
        name = sys.argv[1]
    except IndexError:
        name = 'output'
    main(name)