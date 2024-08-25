#!/bin/bash
#SBATCH -A physics_engg
#SBATCH --ntasks=1
#SBATCH --job-name=clean.up
#SBATCH --output=output_run1/cleanup.out


echo "========= Job started  at `date` on `hostname -s` =========="

python condensor.py

python plotter.py

echo "========= Job finished at `date` =========="

