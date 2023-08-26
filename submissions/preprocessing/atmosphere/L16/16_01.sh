#!/bin/bash -l
#PBS -N coarse_grain_16_01_L16
#PBS -A uclb0017
#PBS -l select=1:ncpus=1:mem=16GB
#PBS -l walltime=08:00:00
#PBS -q casper
#PBS -j oe
#PBS -M fkg2106@columbia.edu

### Run program
module load conda
conda activate npl

python3 preprocessing/atmosphere/coarse_grain_data.py --sim-name=16_01 --output-dir=/glade/u/home/fginnold/master_project/datasets/atmosphere/lz2/lxy16/ --Lx=16 --Ly=16 --Lz=2 --surface-flux=0.01
