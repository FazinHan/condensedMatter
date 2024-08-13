import os

scratch_path = os.path.join('$SCRATCH','condensedMatter')

for roots, dirs, fnames in os.walk('.'):
    for direc in dirs:
        if 'results_version' in direc:
            os.rename(os.path.join(roots, direc), scratch_path)