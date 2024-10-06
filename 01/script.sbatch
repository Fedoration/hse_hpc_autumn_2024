#!/bin/bash
#SBATCH --job-name=falaputin_matrix_mult
#SBATCH --output=output.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=00:15:00

# Модули
module load gcc
module load OpenBlas/v0.3.18
module load INTEL/oneAPI_2022
module load mkl/latest

# Устанавливаем количество потоков для OpenMP
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Компиляция
make
