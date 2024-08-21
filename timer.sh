#!/bin/bash
#SBATCH -A physics_engg
#SBATCH --output=output_run1/timer.out

module load conda

echo "===========FULL SYSTEM SIZE TIMER=========="

echo "Timing get_k_space..."
python -m timeit "from main import get_k_space" "get_k_space()"
echo ""

echo "Timing ft_potential_builder_3..."
python -m timeit "from main import ft_potential_builder_3" "ft_potential_builder_3()"
echo ""

echo "Timing hamiltonian..."
python -m timeit "from main import hamiltonian" "hamiltonian()"
echo ""

echo "Timing conductivity_vectorised..."
python -m timeit "from main import conductivity_vectorised" "conductivity_vectorised()"
echo ""

echo "Timing determine_next_filename..."
python -m timeit "from main import determine_next_filename" "determine_next_filename()"
echo ""