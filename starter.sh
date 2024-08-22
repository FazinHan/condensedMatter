#!/bin/bash
#SBATCH --job-name=tilted.fermion
#SBATCH --output=output_run1/fermion.%a.out
#SBATCH --error=output_run1/fermion.%a.err
#SBATCH --time=4-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=40
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

srun --exclusive -N1 mkdir output_data/results_version/run${runstring[*]}0 || python main.py "${runstring[*]}0"
srun --exclusive -N1 mkdir output_data/results_version/run${runstring[*]}1 || python main.py "${runstring[*]}1"
srun --exclusive -N1 mkdir output_data/results_version/run${runstring[*]}2 || python main.py "${runstring[*]}2"
srun --exclusive -N1 mkdir output_data/results_version/run${runstring[*]}3 || python main.py "${runstring[*]}3"
srun --exclusive -N1 mkdir output_data/results_version/run${runstring[*]}4 || python main.py "${runstring[*]}4"
srun --exclusive -N1 mkdir output_data/results_version/run${runstring[*]}5 || python main.py "${runstring[*]}5"
srun --exclusive -N1 mkdir output_data/results_version/run${runstring[*]}6 || python main.py "${runstring[*]}6"
srun --exclusive -N1 mkdir output_data/results_version/run${runstring[*]}7 || python main.py "${runstring[*]}7"
srun --exclusive -N1 mkdir output_data/results_version/run${runstring[*]}8 || python main.py "${runstring[*]}8"
srun --exclusive -N1 mkdir output_data/results_version/run${runstring[*]}9 || python main.py "${runstring[*]}9"
srun --exclusive -N1 mkdir output_data/results_version/run${runstring[*]}10 || python main.py "${runstring[*]}10"
srun --exclusive -N1 mkdir output_data/results_version/run${runstring[*]}11 || python main.py "${runstring[*]}11"
srun --exclusive -N1 mkdir output_data/results_version/run${runstring[*]}12 || python main.py "${runstring[*]}12"
srun --exclusive -N1 mkdir output_data/results_version/run${runstring[*]}13 || python main.py "${runstring[*]}13"
srun --exclusive -N1 mkdir output_data/results_version/run${runstring[*]}14 || python main.py "${runstring[*]}14"
srun --exclusive -N1 mkdir output_data/results_version/run${runstring[*]}15 || python main.py "${runstring[*]}15"
srun --exclusive -N1 mkdir output_data/results_version/run${runstring[*]}16 || python main.py "${runstring[*]}16"
srun --exclusive -N1 mkdir output_data/results_version/run${runstring[*]}17 || python main.py "${runstring[*]}17"
srun --exclusive -N1 mkdir output_data/results_version/run${runstring[*]}18 || python main.py "${runstring[*]}18"
srun --exclusive -N1 mkdir output_data/results_version/run${runstring[*]}19 || python main.py "${runstring[*]}19"
srun --exclusive -N1 mkdir output_data/results_version/run${runstring[*]}20 || python main.py "${runstring[*]}20"
srun --exclusive -N1 mkdir output_data/results_version/run${runstring[*]}21 || python main.py "${runstring[*]}21"
srun --exclusive -N1 mkdir output_data/results_version/run${runstring[*]}22 || python main.py "${runstring[*]}22"
srun --exclusive -N1 mkdir output_data/results_version/run${runstring[*]}23 || python main.py "${runstring[*]}23"
srun --exclusive -N1 mkdir output_data/results_version/run${runstring[*]}24 || python main.py "${runstring[*]}24"
srun --exclusive -N1 mkdir output_data/results_version/run${runstring[*]}25 || python main.py "${runstring[*]}25"
srun --exclusive -N1 mkdir output_data/results_version/run${runstring[*]}26 || python main.py "${runstring[*]}26"
srun --exclusive -N1 mkdir output_data/results_version/run${runstring[*]}27 || python main.py "${runstring[*]}27"
srun --exclusive -N1 mkdir output_data/results_version/run${runstring[*]}28 || python main.py "${runstring[*]}28"
srun --exclusive -N1 mkdir output_data/results_version/run${runstring[*]}29 || python main.py "${runstring[*]}29"
srun --exclusive -N1 mkdir output_data/results_version/run${runstring[*]}30 || python main.py "${runstring[*]}30"
srun --exclusive -N1 mkdir output_data/results_version/run${runstring[*]}31 || python main.py "${runstring[*]}31"
srun --exclusive -N1 mkdir output_data/results_version/run${runstring[*]}32 || python main.py "${runstring[*]}32"
srun --exclusive -N1 mkdir output_data/results_version/run${runstring[*]}33 || python main.py "${runstring[*]}33"
srun --exclusive -N1 mkdir output_data/results_version/run${runstring[*]}34 || python main.py "${runstring[*]}34"
srun --exclusive -N1 mkdir output_data/results_version/run${runstring[*]}35 || python main.py "${runstring[*]}35"
srun --exclusive -N1 mkdir output_data/results_version/run${runstring[*]}36 || python main.py "${runstring[*]}36"
srun --exclusive -N1 mkdir output_data/results_version/run${runstring[*]}37 || python main.py "${runstring[*]}37"
srun --exclusive -N1 mkdir output_data/results_version/run${runstring[*]}38 || python main.py "${runstring[*]}38"
srun --exclusive -N1 mkdir output_data/results_version/run${runstring[*]}39 || python main.py "${runstring[*]}39"

wait $!

echo "========= Job finished at `date` =========="
