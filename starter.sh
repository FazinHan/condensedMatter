#!/bin/bash
#SBATCH -N 5
#SBATCH --job-name=tilted.fermion
#SBATCH --output=output_run1/fermion.out
#SBATCH --time=04:00:00
#SBATCH --ntasks-per-node=40
#SBATCH -A physics_engg
#SBATCH --mail-user=fizaan.khan.phy21@iitbhu.ac.in

echo "========= Job started  at `date` on `hostname -s` =========="

export I_MPI_HYDRA_TOPOLIB=ipl
export OMP_NUM_THREADS=1

echo "Job id       : $SLURM_JOB_ID"

ntasks=200


mpirun -np ${ntasks} python -O main.py

echo "======== Conductivities finished at `date` ========="

mpirun -np ${ntasks} python condensor.py

python plotter.py

mkdir output_data/results_version

echo "========= Job finished at `date` =========="
