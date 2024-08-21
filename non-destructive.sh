#!/bin/bash
#SBATCH -A physics_engg
#SBATCH --ntasks=1
#SBATCH --job-name=clean.up
#SBATCH --output=output_run1/cleanup.out
#SBATCH --array=1-5%5
#SBATCH --ntasks=40

module load conda

echo "========= Job started  at `date` on `hostname -s` =========="

for num in $(seq 1 $ntasks)

do
srun --exclusive --ntasks=1 python plotter.py save ${runstring[*]}${num}
done

echo "========= Job finished at `date` =========="

