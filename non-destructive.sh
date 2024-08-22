#!/bin/bash
#SBATCH -A physics_engg
#SBATCH --ntasks=1
#SBATCH --job-name=clean.up
#SBATCH --output=output_run1/cleanup.out
#SBATCH --array=1-5
#SBATCH --ntasks=40

module load conda
module load intel/2018.0.1.163
module unload gnu8/8.3.0
source /opt/ohpc/pub/intel2018/compilers_and_libraries_2018.1.163/linux/mkl/bin/mklvars.sh intel64

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

