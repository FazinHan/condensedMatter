#!/bin/bash
#SBATCH -A physics_engg
#SBATCH --ntasks=1
#SBATCH --job-name=clean.up
#SBATCH --output=output_run1/cleanup.out


echo "========= Job started  at `date` on `hostname -s` =========="

python condensor.py

python plotter.py

mv --backup=t output_data/results_version $SCRATCH/data-condensedMatter

echo "data sent to scratch"

mkdir output_data/results_version

echo "========= Job finished at `date` =========="

