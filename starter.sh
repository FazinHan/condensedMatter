#!/bin/bash
#SBATCH --job-name=tilted.fermion
#SBATCH --output=output_run1/fermion.%a.out
#SBATCH --error=output_run1/fermion.%a.err
#SBATCH --time=4-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=40
#SBATCH -A physics_engg
#SBATCH --mem-per-cpu=200M
#SBATCH --array=1-5%5
#SBATCH --mail-user=fizaan.khan.phy21@iitbhu.ac.in

source /opt/ohpc/pub/intel2018/compilers_and_libraries_2018.1.163/linux/mkl/bin/mklvars.sh intel64

echo "========= Job started  at `date` on `hostname -s` =========="

export OMP_NUM_THREADS=1

echo "Array job id : $SLURM_ARRAY_JOB_ID"
echo "Job id       : $SLURM_JOB_ID"
echo "Array task id: $SLURM_ARRAY_TASK_ID"

module load conda

runstring=(`echo ${SLURM_ARRAY_TASK_ID:-1}`)
echo "Track-ID: ${runstring[0]}"

ntasks=$SLURM_NTASKS_PER_NODE

for num in $(seq 1 $ntasks)

do
srun --exclusive -N1 mkdir output_data/results_version/run${runstring[*]}${num} || python main.py "${runstring[*]}${num}"
done

wait $!

echo "========= Job finished at `date` =========="
