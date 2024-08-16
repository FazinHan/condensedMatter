from main import determine_next_filename
import sys, os
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt



def main(save):
    conductivities = []
    dirname = 'output_data'
    version = determine_next_filename('results_version',folder='output_data',direc=True,exists=True)

    directory = version
    
    for root, _, fnames in os.walk(directory):
    # for root, _, fnames in os.walk(dirname):
        for fname in fnames:
            if fname != 'params.txt' and 'py' not in fname:# and 'results_version' not in root:
                with open(os.path.join(root, fname),'r') as file:
                    # print(root,fname)
                    data = eval(file.read())
                    conductivities.append(data)
    conductivities = np.array(conductivities)
    conductivities = np.sum(conductivities, axis=0)
    
    fname = os.path.join(directory, 'run110','length1.npy')
    with open(fname,'rb') as file:
        L = np.load(file)

    # print(conductivities.shape)
    # print(L.shape)

    # print(L.shape)
    cs = CubicSpline(L, conductivities)
    g = cs(L)
    beta = cs(L, 1)
    plotter(L[:-1], np.abs(g)[:-1], np.abs(beta)[:-1], save, folder=dirname)
    # plotter(L, np.abs(g), np.abs(beta), name!='output', name)

if __name__=="__main__":
    try:
        save = sys.argv[1]
    except IndexError:
        save = False
    main(save)