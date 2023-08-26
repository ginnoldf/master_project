#!/bin/bash -l
#PBS -N ocean_train_test_split
#PBS -A uclb0017
#PBS -l select=1:ncpus=16:mem=256GB
#PBS -l walltime=08:00:00
#PBS -q casper
#PBS -j oe
#PBS -M fkg2106@columbia.edu

### Run program
module load conda
conda activate npl

python3 preprocessing/ocean/train_test_split.py
