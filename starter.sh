#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=tilted.fermion
#SBATCH --output=output_run1/fermion.%a.out
#SBATCH --error=output_run1/fermion.%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=40
#SBATCH -A physics_engg
#SBATCH --array=1-5
#SBATCH --mail-user=fizaan.khan.phy21@iitbhu.ac.in

echo "========= Job started  at `date` on `hostname -s` =========="

export I_MPI_HYDRA_TOPOLIB=ipl
export OMP_NUM_THREADS=1

echo "Array job id : $SLURM_ARRAY_JOB_ID"
echo "Job id       : $SLURM_JOB_ID"
echo "Array task id: $SLURM_ARRAY_TASK_ID"

runstring=(`echo ${SLURM_ARRAY_TASK_ID:-1}`)
echo "Track-ID: ${runstring[0]}"

ntasks=40


mpirun -np ${ntasks} python -O main.py "${runstring[*]}"



echo "========= Job finished at `date` =========="
