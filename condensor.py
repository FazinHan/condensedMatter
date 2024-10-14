import os
import numpy as np
import re

def determine_next_filename(fname='output',filetype='png',folder='graphics',direc=False,exists=False):
    num = 1
    if direc:
        filename = lambda num: f'{fname}{num}'
        while os.path.isdir(os.path.join('.',folder,filename(num))):
            num += 1
        else:
            if exists:
                num -= 1
        return os.path.join(folder,filename(num))
    filename = lambda num: f'{fname}{num}.{filetype}'
    while os.path.isfile(os.path.join('.',folder,filename(num))):
        num += 1
    else:
        if exists:
            num -= 1
    return os.path.join(folder,filename(num))

def main():
    conductivities = []

    directory = os.path.join('output_data','results_version')

    for root, _, fnames in os.walk(directory):
        for fname in fnames:
            if 'output' in fname and '.txt' in fname:
                with open(os.path.join(root, fname),'r') as file:

                    julia_list_str = file.read()

                    complex_numbers = re.findall(r'([\d.]+) \+ ([\d.]+)im', julia_list_str)

                    data = [complex(float(real), float(imag)) for real, imag in complex_numbers]

                    conductivities.append(data)
                    #print(data)

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
