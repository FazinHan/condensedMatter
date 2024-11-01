#!/bin/bash
#SBATCH -N 8
#SBATCH --job-name=tilted.fermion
#SBATCH --output=fermion.out
#SBATCH --time=4-00:00:00
#SBATCH --ntasks-per-node=24

echo "========= Job started  at `date` on `hostname -s` =========="

#export I_MPI_HYDRA_TOPOLIB=ipl
#export OMP_NUM_THREADS=1

echo "Job id       : $SLURM_JOB_ID"

export I_MPI_FALLBACK=disable
export I_MPI_FABRICS=shm:tmi
export I_MPI_DEBUG=9 
export OMP_NUM_THREADS=24

source ~/.bashrc

mkdir -p output_data/results_version

time mpiexec.hydra -genv I_MPI_DEBUG 9 -n $SLURM_NTASKS -genv OMP_NUM_THREADS 24 julia df_disorder_new.jl

echo "======== Conductivities finished at `date` ========="

python condensor.py

python plotter.py

mkdir -p output_data/results_version

echo "========= Job finished at `date` =========="
