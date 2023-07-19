#!/bin/bash -l
#PBS -N maml_cnn_dnn2_run4_b_32
#PBS -A uclb0017
#PBS -l select=1:ncpus=4:mem=16GB
#PBS -l walltime=08:00:00
#PBS -q casper
#PBS -j oe
#PBS -M fkg2106@columbia.edu

### Run program
module load conda
conda activate training_cuda
export PYTHONPATH=/glade/u/home/fginnold/master_project

python3 training/main.py --config-file=submissions/training/maml/cnn_dnn2_run4/b_32.yaml