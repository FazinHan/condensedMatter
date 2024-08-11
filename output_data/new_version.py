import os

def determine_next_filename(fname='output',filetype='png',direc=False,exists=False):
    num = 1
    if direc:
        filename = lambda num: f'{fname}{num}'
        while os.path.isdir(os.path.join('.',filename(num))):
            num += 1
        else:
            if exists:
                num -= 1
        return os.path.join(filename(num))
    filename = lambda num: f'{fname}{num}.{filetype}'
    while os.path.isfile(os.path.join('.',filename(num))):
        num += 1
    else:
        if exists:
            num -= 1
    return os.path.join(filename(num))

cwd = os.path.basename(os.getcwd())

if cwd == 'output_data':
    dname = determine_next_filename('results_version',direc=True)
    os.mkdir(dname)
    
    for root, dirs, fnames in os.walk('.'):
        for direc in dirs:
            if direc != '.ipynb_checkpoints' and dname != direc:
                os.rename(direc,os.path.join(root,dname,direc))