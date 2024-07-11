from main import determine_next_filename, plotter
import sys
import numpy as np
import matplotlib.pyplot as plt

def main(name):
    fname = determine_next_filename('data','npz','output data',exists=True)
    with open(fname,'rb') as file:
        data = np.load(file)
        L = data['L']
        eta = data['eta']
        g = data['conductivities']
        beta = data['beta']
    plotter(L, np.abs(g), np.abs(beta), name!='output', name)

if __name__=="__main__":
    try:
        name = sys.argv[1]
    except IndexError:
        name = 'output'
    main(name)