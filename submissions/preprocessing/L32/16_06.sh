#!/bin/bash -l
#PBS -N coarse_grain_16_06_L32
#PBS -A uclb0017
#PBS -l select=1:ncpus=1:mem=16GB
#PBS -l walltime=08:00:00
#PBS -q casper
#PBS -j oe
#PBS -M fkg2106@columbia.edu

### Run program
module load conda
conda activate npl

python3 preprocessing/coarse_grain_data.py --sim_name=16_06 --output_dir=/glade/u/home/fginnold/master_project/data/training_data/L_32/ --Lx=32 --Ly=32
