#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=tilted.fermion
#SBATCH --output=output_run1/fermion.out
#SBATCH --time=04:00:00
#SBATCH --ntasks-per-node=1
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

#time mpiexec.hydra -genv I_MPI_DEBUG 9 -n $SLURM_NTASKS -genv OMP_NUM_THREADS 40 python -O main.py

echo "======== Conductivities finished at `date` ========="

time python condensor.py

python plotter.py

mkdir output_data/results_version

echo "========= Job finished at `date` =========="
