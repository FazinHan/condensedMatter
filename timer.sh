#!/bin/bash
#SBATCH -A physics_engg
#SBATCH --output=output_run1/timer.out
#SBATCH --ntasks=5

module load conda
module load intel/2018.0.1.163
module unload gnu8/8.3.0
source /opt/ohpc/pub/intel2018/compilers_and_libraries_2018.1.163/linux/mkl/bin/mklvars.sh intel64

export I_MPI_FALLBACK=disable
export I_MPI_FABRICS=shm:tmi
export I_MPI_DEBUG=9
export OMP_NUM_THREADS=40

echo "===========FULL SYSTEM SIZE TIMER========="

#mpirun -np 5 echo $(hostname)-$$

counter=0

echo `date`
# Run 40 instances of your Python script in parallel with unique numbers
srun --exclusive -N1 python temp.py
echo `date`
