from main import determine_next_filename
import os
import numpy as np

def main():
    conductivities = []

    directory = os.path.join('output_data','results_version')

    for root, _, fnames in os.walk(directory):
        for fname in fnames:
            if 'output' in fname and '.txt' in fname:
                with open(os.path.join(root, fname),'r') as file:
                    data = eval(file.read())
                    conductivities.append(data)

    return np.array(conductivities)

def run():
    
    results = main()

    lfname = os.path.join('output_data','results_version','length.npy')
    with open(lfname,'rb') as file:
        L = np.load(file)

    conductivities = np.mean(results, axis=0)

    sem = np.std(results, axis=0) / np.sqrt(results.shape[0])

    fname = determine_next_filename(folder=os.path.join('output_data','data'),fname='L_cond_data',filetype='npz')

    with open(fname, 'wb') as file:
        np.savez(file,L=L, conductivities=conductivities, sem=sem)


if __name__=="__main__":
    run()
