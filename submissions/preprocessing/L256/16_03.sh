#!/bin/bash -l
#PBS -N coarse_grain_16_03_L256
#PBS -A uclb0017
#PBS -l select=1:ncpus=1:mem=16GB
#PBS -l walltime=08:00:00
#PBS -q casper
#PBS -j oe
#PBS -M fkg2106@columbia.edu

### Run program
module load conda
conda activate npl

python3 preprocessing/coarse_grain_data.py --sim_name=16_03 --output_dir=/glade/u/home/fginnold/master_project/data/training_data/L_256/ --Lx=256 --Ly=256
