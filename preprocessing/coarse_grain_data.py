import os
import time
import numpy as np

import config

TSCALE = 300


def coarse_grain(data, cf):
    """
    input data has the shape (z-highres, y-highres, x-highres)
    output data has the shape (z-coarse, y-coarse, x-coarse)
    cf is a tuple with the coarsening factors for the box dimensions (z, x, y)
    """
    coarse = tuple(np.divide(data.shape, cf).astype(int))
    out = np.zeros(shape=coarse)

    for idx, _ in np.ndenumerate(out):
        start = np.multiply(idx, cf).astype(int)
        end = start + cf
        out[idx] = np.mean(data[start[0]:end[0], start[1]:end[1], start[2]:end[2]], axis=(0, 1, 2))
    return out


def make_flux(tt, w, t_sgs, cf):
    wt = (coarse_grain(w * tt, cf) - coarse_grain(w, cf) * coarse_grain(tt, cf) + coarse_grain(t_sgs, cf))
    return wt


def shift(data):
    zmax = data.shape[0]
    dd = np.copy(data)
    for zz in range(1, zmax):
        dd[zz, :, :] = 0.5 * (data[zz - 1, :, :] + data[zz, :, :])
    return dd


def make_tke(w, u, v, cf):
    uu = coarse_grain(u * u, cf) - coarse_grain(u, cf) * coarse_grain(u, cf)
    vv = coarse_grain(v * v, cf) - coarse_grain(v, cf) * coarse_grain(v, cf)
    ww = coarse_grain(w * w, cf) - coarse_grain(w, cf) * coarse_grain(w, cf)
    return 0.5 * (ww + uu + vv)


def make_filelist(search_dir, search_for, file_ending):
    # get all matching files with the correct ending
    directory = os.listdir(search_dir)
    files = []
    for file in directory:
        if file.endswith(file_ending) and search_for in file:
            files.append(file[16:-4])
    return files


def main():
    args = config.Config()

    # get sorted input file list
    file_list = make_filelist(search_dir=args.input_dir.name, search_for='uins', file_ending='.out')
    file_list = sorted(file_list, key=lambda x: int(x))

    # initialize numpy arrays
    num_timesteps = len(file_list)
    theta_coarse = np.zeros(shape=(num_timesteps,) + args.coarse_shape)
    tkes_coarse = np.zeros(shape=(num_timesteps,) + args.coarse_shape)
    turb_heat_flux_coarse = np.zeros(shape=(num_timesteps,) + args.coarse_shape)

    # logging
    print("We will handle " + str(num_timesteps) + " timesteps.")

    # iterate over given timesteps from LES
    for timestep_idx, timestep in enumerate(file_list):
        tic = time.perf_counter()

        # constructing the name of the inputfile (LES output)
        fileinst_w = "aver_wins" + str(timestep) + str(timestep) + '.out'
        fileinst_u = "aver_uins" + str(timestep) + str(timestep) + '.out'
        fileinst_v = "aver_vins" + str(timestep) + str(timestep) + '.out'
        fileinst_theta = "aver_thetains" + str(timestep) + str(timestep) + '.out'
        filesgs_t = "aver_sgs_t30" + str(timestep[1:]) + ".out"

        # reading data to numpy arrays
        w_velocity = np.loadtxt((args.input_dir / fileinst_w).name).reshape((256, 256, 257))
        u_velocity = np.loadtxt((args.input_dir / fileinst_u).name).reshape((256, 256, 257))
        v_velocity = np.loadtxt((args.input_dir / fileinst_v).name).reshape((256, 256, 257))
        theta = np.loadtxt((args.input_dir / fileinst_theta).name).reshape((256, 256, 257))
        t_sgs = np.loadtxt((args.input_dir / filesgs_t).name).reshape((256, 256, 257))

        # removing first column. It contains time information
        w_velocity = w_velocity[:, :, 1:]
        u_velocity = shift(u_velocity[:, :, 1:])
        v_velocity = shift(v_velocity[:, :, 1:])
        t_sgs = t_sgs[:, :, 1:]
        theta = theta[:, :, 1:]

        theta = shift(theta) * TSCALE

        # calculate theta, tke and theta covariance into existing numpy array
        theta_coarse[timestep_idx] = coarse_grain(theta, cf=args.cf)
        tkes_coarse[timestep_idx] = make_tke(w=w_velocity, u=u_velocity, v=v_velocity, cf=args.cf)
        turb_heat_flux_coarse[timestep_idx] = make_flux(tt=theta, w=w_velocity, t_sgs=t_sgs, cf=args.cf)

        # logging
        toc = time.perf_counter()
        print("Calculation done for timestep " + str(timestep_idx) + f" in {toc - tic:0.4f}s.")

    # save constructed arrays in files
    np.save((args.output_dir / args.sim_name / 'theta').name, theta_coarse)
    np.save((args.output_dir / args.sim_name / 'tkes').name, tkes_coarse)
    np.save((args.output_dir / args.sim_name / 'turb_heat_flux').name, turb_heat_flux_coarse)


if __name__ == '__main__':
    main()
