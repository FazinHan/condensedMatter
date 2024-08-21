from main import determine_next_filename
import sys, os
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

def plotter(L, conductivities, beta, save, folder=''):
    fig, axs = plt.subplots(2,1)
    
    axs[1].plot(L, conductivities,'.')
    axs[0].plot(conductivities, beta,'.')
    axs[1].set_xlabel('L')
    axs[1].set_ylabel('g')
    axs[0].set_xlabel('g')
    axs[0].set_ylabel('$\\beta$')
    fig.tight_layout()
    name = determine_next_filename('plot', folder=folder, filetype='png')#fname='eta_variance')
    if save:
        plt.savefig(name)
        print('plotted to',name)
        os.rename(os.path.join(folder,'params.txt'), name.strip('.png')+'_params.txt')
        print('parameter file renamed to',name.strip('.png')+'_params.txt')
    else:
        plt.show()

def main(save):
    conductivities = []

    directory = os.path.join('output_data','results_version','run'+sys.argv[2])
    
    for root, _, fnames in os.walk(directory):
        for fname in fnames:
	    if 'npy' in fname:	
		lfname = fname
            if 'params.txt' not in fname and 'txt' in fname:# and 'results_version' not in root:
                with open(os.path.join(root, fname),'r') as file:
                    data = eval(file.read())
                    conductivities.append(data)
    conductivities = np.array(conductivities)
    conductivities = np.sum(conductivities, axis=0)
    
    #lfname = os.path.join(directory,'length1.npy')
    with open(lfname,'rb') as file:
        L = np.load(file)

    cs = CubicSpline(np.log(L), np.log(conductivities))
    beta = cs(np.log(L), 1)
    plotter(L, conductivities, beta, save, folder=directory)

if __name__=="__main__":
    try:
        save = sys.argv[1]
    except IndexError:
        save = False
    main(save)
