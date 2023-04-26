import os
import numpy as np

TSCALE = 300


def coarse_grain(datain):
    return np.mean(datain, axis=(1, 2))


def make_mean(tt):
    print("shape of tt: " + str(np.shape(tt)))
    tt_mean = np.mean(tt, axis=(1, 2))#.reshape(1, -1, 1)
    print("shape of np.mean(tt, axis=(1, 2)): " + str(np.shape(np.mean(tt, axis=(1, 2)))))
    print("shape of tt_mean: " + str(np.shape(tt_mean)))
    return tt_mean


def make_flux(tt, w, t_sgs):
    wt = (coarse_grain(w * tt) - coarse_grain(w) * coarse_grain(tt) + coarse_grain(t_sgs)).reshape(1, -1, 1)
    return wt


def shift(data):
    zmax = data.shape[0]
    dd = np.copy(data)
    for zz in range(1, zmax):
        dd[zz, :, :] = 0.5 * (data[zz - 1, :, :] + data[zz, :, :])
    return dd


def make_tke(w, u, v):
    uu = coarse_grain(u * u) - coarse_grain(u) * coarse_grain(u)
    vv = coarse_grain(v * v) - coarse_grain(v) * coarse_grain(v)
    ww = coarse_grain(w * w) - coarse_grain(w) * coarse_grain(w)

    W = coarse_grain(w)
    V = coarse_grain(v)
    U = coarse_grain(u)

    partu = coarse_grain(w * u * u) - W * U * U - W * (coarse_grain(u * u) - U * U) \
            - 2 * U * (coarse_grain(u * w) - U * W)
    partv = coarse_grain(w * v * v) - W * V * V - W * (coarse_grain(v * v) - V * V) \
            - 2 * V * (coarse_grain(v * w) - V * W)
    partw = coarse_grain(w * w * w) - W * W * W - W * (coarse_grain(w * w) - W * W) \
            - 2 * W * (coarse_grain(w * w) - W * W)
    we = partu + partv + partw

    return 0.5 * (ww + uu + vv).reshape(1, -1), ww.reshape(1, -1), 0.5 * we.reshape(1, -1)


def make_filelist(search_dir, search_for, file_ending):
    # get all matching files with the correct ending
    directory = os.listdir(search_dir)
    files = []
    for file in directory:
        if file.endswith(file_ending) and search_for in file:
            files.append(file[16:-4])
    return files


def main():
    input_dir = '/glade/scratch/sshamekh/'
    input_prefix = 'cbl_'
    output_dir = '/glade/u/home/fginnold/training_data/'
    sim_name = '16_01'

    # get sorted input file list
    path = input_dir + input_prefix + sim_name + '/output/'
    file_list = make_filelist(search_dir=path, search_for='uins', file_ending='.out')
    file_list = sorted(file_list, key=lambda x: int(x))

    # initialize numpy arrays
    num_timesteps = len(file_list)
    num_vertical_levels = 256
    scalar_mean = np.zeros(shape=(num_timesteps, num_vertical_levels))
    tkes = np.zeros(shape=(num_timesteps, num_vertical_levels))
    all_flux = np.zeros(shape=(num_timesteps, num_vertical_levels))

    # logging
    print("We will handle " + str(num_timesteps) + " timesteps.")

    # iterate over given timesteps from LES
    for timestep_counter in range(num_timesteps):
        timestep = file_list[timestep_counter]

        # constructing the name of the inputfile (LES output)
        fileinst_w = "aver_wins" + str(timestep) + str(timestep) + '.out'
        fileinst_u = "aver_uins" + str(timestep) + str(timestep) + '.out'
        fileinst_v = "aver_vins" + str(timestep) + str(timestep) + '.out'
        fileinst_theta = "aver_thetains" + str(timestep) + str(timestep) + '.out'
        filesgs_t = "aver_sgs_t30" + str(timestep[1:]) + ".out"

        # reading data to numpy arrays
        w_velocity = np.loadtxt(path + fileinst_w).reshape((256, 256, 257))
        u_velocity = np.loadtxt(path + fileinst_u).reshape((256, 256, 257))
        v_velocity = np.loadtxt(path + fileinst_v).reshape((256, 256, 257))
        theta = np.loadtxt(path + fileinst_theta).reshape((256, 256, 257))
        t_sgs = np.loadtxt(path + filesgs_t).reshape((256, 256, 257))

        # removing first column. It contains time information
        w_velocity = w_velocity[:, :, 1:]
        u_velocity = shift(u_velocity[:, :, 1:])
        v_velocity = shift(v_velocity[:, :, 1:])
        t_sgs = t_sgs[:, :, 1:]
        theta = theta[:, :, 1:]

        theta = shift(theta) * TSCALE

        # calculate theta, tke and theta covariance into existing numpy array
        scalar_mean[timestep_counter] = make_mean(theta)
        tkes[timestep_counter] = make_tke(w_velocity, u_velocity, v_velocity)
        all_flux[timestep_counter] = make_flux(theta, w_velocity, t_sgs)

        # logging
        print("Calculation done for timestep " + str(timestep_counter) + " .")

    # save constructed arrays in files
    np.save(output_dir + sim_name + '_v3' + '_scalar_mean', scalar_mean)
    np.save(output_dir + sim_name + '_v3' + '_tkes', tkes)
    np.save(output_dir + sim_name + '_v3' + '_fluxes', np.array(all_flux))


if __name__ == '__main__':
    main()
