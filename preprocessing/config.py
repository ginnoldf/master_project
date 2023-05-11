import argparse
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser(description='Args for coarse_grain.py')

parser.add_argument('--sim_dir', default='/glade/scratch/sshamekh/')
parser.add_argument('--input_prefix', default='cbl_')
parser.add_argument('--sim_name')
parser.add_argument('--output_dir')
parser.add_argument('--input_dim_z', type=int, default=256)
parser.add_argument('--input_dim_x', type=int, default=256)
parser.add_argument('--input_dim_y', type=int, default=256)
parser.add_argument('--Lz', type=int, default=1)
parser.add_argument('--Lx', type=int, default=256)
parser.add_argument('--Ly', type=int, default=256)

args = parser.parse_args()


class Config:
    input_dir = Path(args.simdir) / args.input_prefix + args.sim_name / 'output'
    output_dir = Path(args.output_dir)
    sim_name = args.sim_name
    input_dim_z = args.input_dim_z
    input_dim_x = args.input_dim_x
    input_dim_y = args.input_dim_y
    lx = args.Lx
    ly = args.Ly
    lz = args.Lz

    # coarsening factor
    cf = (lz, lx, ly)

    # input shape
    input_shape = (input_dim_z, input_dim_x, input_dim_y)

    # coarse shape
    coarse_shape = tuple(np.divide(input_shape, cf).astype(int))
