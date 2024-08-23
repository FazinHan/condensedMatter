#!/bin/bash
#SBATCH -A physics_engg
#SBATCH --ntasks=1
#SBATCH --job-name=clean.up
#SBATCH --output=output_run1/cleanup.out
#SBATCH --array=1-5
#SBATCH --ntasks=40

module load conda

export I_MPI_FALLBACK=disable
export I_MPI_FABRICS=shm:tmi
export I_MPI_DEBUG=9
export OMP_NUM_THREADS=40

runstring=(`echo ${SLURM_ARRAY_TASK_ID:-1}`)

echo "========= Job started  at `date` on `hostname -s` =========="
ntasks=$SLURM_NTASKS

for num in $(seq 1 $ntasks)

do
srun --exclusive -N1 python condensor.py "${runstring[*]}${num}"
done

echo "========= Job finished at `date` =========="

