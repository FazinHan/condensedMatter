from main import determine_next_filename
import sys, os
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

def plotter(L, conductivities, beta, save, name='output', folder=''):
    fig, axs = plt.subplots(2,1)
    # t2 = time.perf_counter()
    # print(f'parallelised: {np.round(t2-t1,3)}s taken')
    
    axs[1].plot(L, conductivities,'.')
    axs[0].plot(conductivities, beta,'.')
    # axs[0].set_xlabel('$\\eta$')
    axs[1].set_xlabel('$\\eta$')
    axs[1].set_ylabel('g')
    axs[0].set_xlabel('g')
    axs[0].set_ylabel('$\\beta$')
    # if name=='output':    
    #     fig.suptitle(randomness.__name__[:-8])
    # else:
    #     fig.suptitle(name)
    fig.tight_layout()
    name = determine_next_filename('plot', folder=folder, filetype='png')#fname='eta_variance')
    if save:
        plt.savefig(name)
        print('plotted to',name)
        os.rename(os.path.join('output_data','params.txt'), name.split('.')[0]+'_params.txt')
        print('parameter file renamed to',name.split('.')[0]+'_params.txt')
    else:
        plt.show()

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