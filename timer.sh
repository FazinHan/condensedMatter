#!/bin/bash

#SBATCH -A physics_engg
#SBATCH --output=out_timer.out

module load conda

python -c "import timeit; from main import conductivity_vectorised; timer = timeit.default_timer; start = timer(); conductivity_vectorised(); print(timer() - start, 's')"
