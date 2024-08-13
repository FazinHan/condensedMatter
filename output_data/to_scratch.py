import os

for roots, dirs, fnames in os.walk('.'):
    for direc in dirs:
        if 'results_version' in direc:
            path = os.path.join(roots, direc)
            os.system(f'mv --backup=t {path} $SCRATCH/condensedMatter')
