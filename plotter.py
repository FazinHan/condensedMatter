from main import determine_next_filename, plotter
import sys, os
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

def main(name):
    conductivities = []
    dirname = determine_next_filename(fname='run',folder='output_data',direc=True,exists=True)
    # fname = determine_next_filename('output','npy','output_data',exists=True)
    
    for _, _, fnames in os.walk(dirname):
        for fname in fnames:
            if 'length' not in fname:
                with open(os.path.join(dirname, fname),'rb') as file:
                    conductivities.append(np.load(file))
    conductivities = np.array(conductivities)
    conductivities = np.sum(conductivities, axis=0)
    
    fname = determine_next_filename(fname='length',folder=dirname, filetype='npy',exists=True)
    with open(fname,'rb') as file:
        L = np.load(file)

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