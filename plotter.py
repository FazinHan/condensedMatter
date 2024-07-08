from main import determine_next_filename, plotter
import sys
import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":
    name = determine_next_filename('data','npz','output data',exists=True)
    with open(name,'rb') as file:
        data = np.load(file)
        L = data['L']
        eta = data['eta']
        g = data['conductivities']
        beta = data['beta']
    try:
        save = int(sys.argv[1])
    except IndexError:
        save = 0
    plotter(L, np.abs(g), np.abs(beta), save)
