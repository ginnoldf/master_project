#!/bin/bash -l
#PBS -N train_simple_lz1_lxy256
#PBS -A uclb0017
#PBS -l select=1:ncpus=1:mem=16GB
#PBS -l walltime=08:00:00
#PBS -q casper
#PBS -j oe
#PBS -M fkg2106@columbia.edu

### Run program
module load conda
conda activate npl
conda env list

python3 training/main.py --config-file=runs/lz1_lxy256.yaml
