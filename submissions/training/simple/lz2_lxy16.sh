#!/bin/bash -l
#PBS -N train_simple_lz2_lxy256
#PBS -A uclb0017
#PBS -l select=1:ncpus=1:mem=16GB
#PBS -l walltime=08:00:00
#PBS -q casper
#PBS -j oe
#PBS -M fkg2106@columbia.edu

### Run program
module load conda
conda activate training
export PYTHONPATH=/glade/u/home/fginnold/master_project

python3 training/main.py --config-file=submissions/training/simple/lz2_lxy16.yaml
