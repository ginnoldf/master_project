import argparse
import numpy as np
from pathlib import Path
import os


def get_args():
    parser = argparse.ArgumentParser(description='Args for coarse_grain.py')

    parser.add_argument('--sim-dir', default='/glade/scratch/sshamekh/')
    parser.add_argument('--input-prefix', default='cbl_')
    parser.add_argument('--sim-name')
    parser.add_argument('--output-dir')
    parser.add_argument('--input-dim-z', type=int, default=256)
    parser.add_argument('--input-dim-x', type=int, default=256)
    parser.add_argument('--input_dim_y', type=int, default=256)
    parser.add_argument('--Lz', type=int, default=1)
    parser.add_argument('--Lx', type=int, default=256)
    parser.add_argument('--Ly', type=int, default=256)
    parser.add_argument('--surface-flux', type=float)
    parser.add_argument('--num-layers', type=int, default=180)

    return parser.parse_args()


class Config:
    args = get_args()
    input_dir = os.path.join(args.sim_dir, args.input_prefix + args.sim_name, 'output')

    # create dirs if necessary
    output_dir = args.output_dir
    sim_name = args.sim_name
    if not os.path.exists(os.path.join(output_dir, sim_name)):
        os.makedirs(os.path.join(output_dir, sim_name))

    input_dim_z = args.input_dim_z
    input_dim_x = args.input_dim_x
    input_dim_y = args.input_dim_y
    lx = args.Lx
    ly = args.Ly
    lz = args.Lz
    surface_flux = args.surface_flux
    num_layers = args.num_layers

    # coarsening factor
    cf = (lz, lx, ly)

    # input shape
    input_shape = (min(input_dim_z, num_layers), input_dim_x, input_dim_y)

    # coarse shape
    coarse_shape = tuple(np.divide(input_shape, cf).astype(int))
