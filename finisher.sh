#!/bin/bash
#SBATCH -A physics_engg
#SBATCH --ntasks=1
#SBATCH --job-name=clean.up
#SBATCH --output=output_run1/cleanup.out

module load conda
module load git_2.41

echo "========= Job started  at `date` on `hostname -s` =========="

cd output_data

python new_version.py

echo "Data moved into directory"

cd ..

python plotter.py save

echo "plot created"

cd output_data

python to_scratch.py

echo "data sent to scratch"

eval "$(ssh-agent -s)" &&  ssh-add ~/.ssh/id_ed25519 && git add . && git commit -m "update" && git push

echo "========= Job finished at `date` =========="

