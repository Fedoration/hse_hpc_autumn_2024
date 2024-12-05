#!/bin/bash
#SBATCH --job-name=falaputin_openmpi
#SBATCH --output=output.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00

# Модули
module load openmpi/4.1.4                       

# Компиляция
make clean
make
