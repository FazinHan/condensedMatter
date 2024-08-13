import os

scratch_path = os.path.join('$SCRATCH','condensedMatter')

for roots, dirs, fnames in os.walk('.'):
    for direc in dirs:
        if 'results_version' in direc:
            path = os.path.join(roots, direc)
            os.system(f'mv {path} {scratch_path}')