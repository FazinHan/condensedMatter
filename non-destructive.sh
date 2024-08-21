#!/bin/bash
#SBATCH -A physics_engg
#SBATCH --ntasks=1
#SBATCH --job-name=clean.up
#SBATCH --output=output_run1/cleanup.out
#SBATCH --array=1-5%5
#SBATCH --ntasks=40

module load conda
module load intel/2018.0.1.163
module unload gnu8/8.3.0
source /opt/ohpc/pub/intel2018/compilers_and_libraries_2018.1.163/linux/mkl/bin/mklvars.sh intel64

export I_MPI_FALLBACK=disable
export I_MPI_FABRICS=shm:tmi
export I_MPI_DEBUG=9
export OMP_NUM_THREADS=40

echo "========= Job started  at `date` on `hostname -s` =========="

mpirun -np 40 python plotter.py save $OMPI_COMM_WORLD_RANK

echo "========= Job finished at `date` =========="

