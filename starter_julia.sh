#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=tilted.fermion
#SBATCH --output=fermion.out
#SBATCH --time=4-00:00:00
#SBATCH --ntasks-per-node=4

echo "========= Job started  at `date` on `hostname -s` =========="

#export I_MPI_HYDRA_TOPOLIB=ipl
#export OMP_NUM_THREADS=1

echo "Job id       : $SLURM_JOB_ID"

source ~/.bashrc
conda init
conda activate tilted-df

export I_MPI_FALLBACK=disable
export I_MPI_FABRICS=shm:tmi
export I_MPI_DEBUG=9 
export OMP_NUM_THREADS=24

mkdir output_data/results_version

time mpiexec.hydra -genv I_MPI_DEBUG 9 -n $SLURM_NTASKS -genv OMP_NUM_THREADS 40 julia julia-main.jl
# time mpiexec.hydra -genv I_MPI_DEBUG 9 -n $SLURM_NTASKS -genv OMP_NUM_THREADS 40 python main.py

echo "======== Conductivities finished at `date` ========="

python condensor.py

python plotter.py

echo "========= Job finished at `date` =========="

