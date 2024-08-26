from main import determine_next_filename
import sys, os
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

def plotter(L, conductivities, beta, data_name, folder=''):
    fig, axs = plt.subplots(2,1)
    
    axs[1].plot(L, conductivities,'.')
    axs[0].plot(conductivities, beta,'.')
    axs[1].set_xlabel('L')
    axs[1].set_ylabel('g')
    axs[0].set_xlabel('g')
    axs[0].set_ylabel('$\\beta$')
    fig.tight_layout()
    name = 'plot_'+os.path.splitext(os.path.basename(data_name))[0]
    plt.savefig(os.path.join(folder,name+'.png'))
    print('plotted to',name+'.png')
    param_name = os.path.join(folder, name+'_params.txt')
    os.rename(os.path.join(folder,'results_version','params.txt'),param_name)
    print('parameter file renamed to',param_name)

def main():
    fname = determine_next_filename(folder=os.path.join('output_data','data'),fname='L_cond_data',filetype='npz',exists=True)

    with open(fname, 'rb') as file:
        data = np.load(file)
        L = data['L']
        conductivities = data['conductivities']

    cs = CubicSpline(np.log(L), np.log(conductivities))
    beta = cs(np.log(L), 1)
    plotter(L, conductivities, beta, fname, folder='output_data')

if __name__=="__main__":
    main()
