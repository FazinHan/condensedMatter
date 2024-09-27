#!/bin/bash
#SBATCH -N 5
#SBATCH --job-name=tilted.fermion
#SBATCH --output=output_run1/fermion.out
#SBATCH --time=4-00:00:00
#SBATCH --ntasks-per-node=40
#SBATCH -A physics_engg
#SBATCH --mail-user=fizaan.khan.phy21@iitbhu.ac.in

echo "========= Job started  at `date` on `hostname -s` =========="

#export I_MPI_HYDRA_TOPOLIB=ipl
#export OMP_NUM_THREADS=1

echo "Job id       : $SLURM_JOB_ID"

export I_MPI_FALLBACK=disable
export I_MPI_FABRICS=shm:tmi
export I_MPI_DEBUG=9 
export OMP_NUM_THREADS=40

source ~/.bashrc

time mpiexec.hydra -genv I_MPI_DEBUG 9 -n $SLURM_NTASKS -genv OMP_NUM_THREADS 40 /scratch/fizaank.phy21.iitbhu/julia-1.10.5/bin/julia julia-main.jl

echo "======== Conductivities finished at `date` ========="

python condensor.py

python plotter.py

mkdir output_data/results_version

push

echo "========= Job finished at `date` =========="
