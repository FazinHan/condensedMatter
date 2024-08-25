#!/bin/bash
#SBATCH --job-name=tilted.fermion
#SBATCH --output=output_run1/fermion.%a.out
#SBATCH --error=output_run1/fermion.%a.err
#SBATCH --time=4-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=40
#SBATCH -A physics_engg
#SBATCH --mem-per-cpu=200M
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

ntasks=$SLURM_NTASKS

for num in $(seq 1 $ntasks)

do
srun --exclusive -N1 --ntasks=1 python main.py "${runstring[*]}${num}"
done


echo "========= Job finished at `date` =========="
