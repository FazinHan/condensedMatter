#!/bin/bash
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=shivay_tests
#SBATCH --output=output_run1/shivay_tests.out

mpirun -np 2 python shivay_tests.py