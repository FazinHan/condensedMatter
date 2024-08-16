#!/bin/bash
#SBATCH -A physics_engg
#SBATCH --ntasks=1
#SBATCH --job-name=clean.up
#SBATCH --output=output_run1/cleanup.out
#SBATCH --array=1-5%5
#SBATCH --ntasks=40

module load conda
module load git_2.41

echo "========= Job started  at `date` on `hostname -s` =========="

for num in $(seq 1 $ntasks)

do
srun --exclusive --ntasks=1 python plotter.py save ${runstring[*]}${num}
done

mv --backup=t output_data/results_version $SCRATCH/data-condensedMatter

echo "data sent to scratch"

mkdir output_data/results_version

echo "========= Job finished at `date` =========="

