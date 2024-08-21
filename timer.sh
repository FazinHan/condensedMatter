#!/bin/bash
#SBATCH -A physics_engg
#SBATCH --output=output_run1/timer.out
#SBATCH --ntasks=5

module load conda

echo "===========FULL SYSTEM SIZE TIMER=========="

parallel -j 5 python -m timeit "print('timing {}...)" "from main import {}" "{}()" ::: "get_k_space" "ft_potential_builder_3" "hamiltonian" "conductivity_vectorised" "determine_next_filename"

