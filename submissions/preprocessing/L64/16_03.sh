#!/bin/bash -l
#PBS -N coarse_grain_16_03_L64
#PBS -A uclb0017
#PBS -l select=1:ncpus=1:mem=16GB
#PBS -l walltime=08:00:00
#PBS -q casper
#PBS -j oe
#PBS -M fkg2106@columbia.edu

### Run program
module load conda
conda activate npl

python3 preprocessing/coarse_grain_data.py --sim-name=16_03 --output-dir=/glade/u/home/fginnold/master_project/datasets/training_data/lz2/lxy64/ --Lx=64 --Ly=64 --Lz=2 --surface-flux=0.03
