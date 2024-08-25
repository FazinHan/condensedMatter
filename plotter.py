from main import determine_next_filename
import sys, os
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

def plotter(L, conductivities, beta, folder=''):
    fig, axs = plt.subplots(2,1)
    
    axs[1].plot(L, conductivities,'.')
    axs[0].plot(conductivities, beta,'.')
    axs[1].set_xlabel('L')
    axs[1].set_ylabel('g')
    axs[0].set_xlabel('g')
    axs[0].set_ylabel('$\\beta$')
    fig.tight_layout()
    name = determine_next_filename('plot', folder=folder, filetype='png')#fname='eta_variance')
    plt.savefig(name)
    print('plotted to',name)
    os.rename(os.path.join(folder,'results_version','params.txt'), name.strip('.png')+'_params.txt')
    print('parameter file renamed to',name.strip('.png')+'_params.txt')

def main():
    fname = determine_next_filename(folder=os.path.join('output_data','data'),fname='L_cond_data',filetype='npz',exists=True)

    with open(fname, 'rb') as file:
        data = np.load(file)
        L = data['L']
        conductivities = data['conductivities']

    cs = CubicSpline(np.log(L), np.log(conductivities))
    beta = cs(np.log(L), 1)
    plotter(L, conductivities, beta, folder='output_data')

if __name__=="__main__":
    main()
