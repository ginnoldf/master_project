#!/bin/bash -l
#PBS -N atm_opt_cos_lz2_lxy32
#PBS -A uclb0017
#PBS -l select=1:ncpus=4:mem=16GB:ngpus=1
#PBS -l walltime=08:00:00
#PBS -q casper
#PBS -j oe
#PBS -M fkg2106@columbia.edu

### Run program
module load conda
conda activate training_cuda
export PYTHONPATH=/glade/u/home/fginnold/master_project
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/glade/work/fginnold/conda-envs/training_cuda/lib

python3 training/main.py --config-file=submissions/training/atmosphere/opt/cosine_annealing/lz2_lxy32.yaml