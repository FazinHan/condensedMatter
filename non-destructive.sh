#!/bin/bash
#SBATCH -N 5
#SBATCH -A physics_engg
#SBATCH --ntasks-per-node=40
#SBATCH --job-name=clean.up
#SBATCH --output=output_run1/cleanup.out

cd $SCRATCH/condensedMatter

echo "========= Job started  at `date` on `hostname -s` =========="

mpirun -np 200 python condensor.py

python plotter.py

mv --backup=t output_data/results_version $SCRATCH/data-condensedMatter

echo "data sent to scratch"

mkdir output_data/results_version

echo "========= Job finished at `date` =========="


