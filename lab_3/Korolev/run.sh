#!/bin/bash
#SBATCH --job-name=mpi_matrix
#SBATCH --time=0:10:00
#SBATCH --ntasks-per-node=12
#SBATCH --partition batch

module load intel/mpi4
./script.sh