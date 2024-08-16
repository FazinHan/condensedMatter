#!/bin/bash
#SBATCH -A physics_engg
#SBATCH --ntasks=1
#SBATCH --job-name=clean.up
#SBATCH --output=output_run1/cleanup.out

module load conda
module load git_2.41

echo "========= Job started  at `date` on `hostname -s` =========="

python plotter.py save

mv --backup=t output_data/results_version $SCRATCH/data-condensedMatter

echo "data sent to scratch"

echo "========= Job finished at `date` =========="

