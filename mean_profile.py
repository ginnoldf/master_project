import os
import numpy as np

# from netCDF4 import Dataset

pi = 3.1415927

Nx = nx = 256
Ny = ny = 256
Nz = nz = 256

dz = 6
dx = 2 * pi * 1000 / nx
dy = 2 * pi * 1000 / ny
dt = 0.025

ustar = 1
Tscale = 300
zo = 0.01
resize_fact = 64
crop = int(resize_fact / 4)
idxy_shift = np.concatenate((np.arange(nx - crop, nx), np.arange(0, nx - crop)))
idxy = np.arange(0, nx)
shiftxy_i = [idxy_shift, None, idxy_shift]
shiftxy_j = [None, idxy_shift, idxy_shift]


def read_ps(var):
    with open(filepath) as f:
        lines = f.readlines()
    ff = np.zeros((256 * y, x))
    for i in range(256 * y):
        for j in range(x):
            if 'E' in (lines[i][19 * j:19 * (j + 1)]):
                ff[i, j] = float(lines[i][19 * j:19 * (j + 1)])
            else:
                ff[i, j] = 0

    return ff[:, :, 1:, :]


def Coarse_grain(datain):
    '''
    input data has the shape (t,z,y-highres,x-highres)
    output data has the shape (z, y-coarse, x-coarse)
    resize factor is coarsening ractor.
    imprortant : resize factor has to be assigned globally!
    '''
    z_len = datain.shape[0]  # grids in vertical
    h_leny = datain.shape[1]  # grids in horizontal
    h_lenx = datain.shape[2]  # grids in horizontal
    lcoarse_x = int(h_lenx / resize_fact)
    lcoarse_y = int(h_leny / resize_fact)  # grids in coase horisontal
    return np.mean(datain, axis=(1, 2))


def avg_percentile(data, perc):
    prc = np.percentile(data, perc, axis=(1, 2), keepdims=True)

    mask = data >= prc
    if perc == 5:
        mask = data <= prc
    return (np.sum(data * mask, axis=(1, 2)) / np.sum(mask, axis=(1, 2))).reshape(1, -1, 1)


def all_percentile(tt, ss, ss2, ss3, w, perc):
    p_t = avg_percentile(tt, perc)
    p_s = avg_percentile(ss, 5)
    p_s2 = avg_percentile(ss2, perc)
    p_s3 = avg_percentile(ss3, 5)
    p_w = avg_percentile(w, perc)
    return np.concatenate((p_t, p_s, p_s2, p_s3, p_w), axis=-1)


def make_mean(tt, ss, ss2, ss3):
    tt_mean = np.mean(tt, axis=(1, 2)).reshape(1, -1, 1)
    ss_mean = np.mean(ss, axis=(1, 2)).reshape(1, -1, 1)
    ss2_mean = np.mean(ss2, axis=(1, 2)).reshape(1, -1, 1)
    ss3_mean = np.mean(ss3, axis=(1, 2)).reshape(1, -1, 1)
    return np.concatenate((tt_mean, ss_mean, ss2_mean, ss3_mean), axis=-1)


def make_flux(tt, ss, ss2, ss3, w, t_sgs, s2_sgs, s3_sgs):
    wt = (Coarse_grain(w * tt) - Coarse_grain(w) * Coarse_grain(tt) + Coarse_grain(t_sgs)).reshape(1, -1, 1)
    ws = (Coarse_grain(w * ss) - Coarse_grain(w) * Coarse_grain(ss)).reshape(1, -1, 1)
    ws2 = (Coarse_grain(w * ss2) - Coarse_grain(w) * Coarse_grain(ss2) + Coarse_grain(s2_sgs)).reshape(1, -1, 1)
    ws3 = (Coarse_grain(w * ss3) - Coarse_grain(w) * Coarse_grain(ss3) + + Coarse_grain(s3_sgs)).reshape(1, -1, 1)
    return np.concatenate((wt, ws, ws2, ws3), axis=-1)


def compute_second_moment(aa, bb):
    sigma = Coarse_grain(aa * bb) - Coarse_grain(aa) * Coarse_grain(bb)
    return sigma.reshape(1, -1)


def compute_third_moment(aa, bb, cc):
    firstterm = Coarse_grain(aa * bb * cc) - Coarse_grain(aa) * Coarse_grain(bb) * Coarse_grain(cc)
    secondterm = Coarse_grain(cc) * (Coarse_grain(aa * bb) - Coarse_grain(aa) * Coarse_grain(bb))
    thirdterm = Coarse_grain(aa) * (Coarse_grain(cc * bb) - Coarse_grain(cc) * Coarse_grain(bb))
    fourthterm = Coarse_grain(bb) * (Coarse_grain(cc * aa) - Coarse_grain(aa) * Coarse_grain(cc))

    return (firstterm - secondterm - thirdterm - fourthterm).reshape(1, -1)


def concat(ds1, ds2):
    for i in range(len(ds1)):
        ds1[i] = np.concatenate((ds1[i], ds2[i]), axis=-1)
    return ds1


def shift(data):
    zmax = data.shape[0]
    dd = np.copy(data)
    for zz in range(1, zmax):
        dd[zz, :, :] = 0.5 * (data[zz - 1, :, :] + data[zz, :, :])
    return dd


def make_tke(w, u, v):
    uu = Coarse_grain(u * u) - Coarse_grain(u) * Coarse_grain(u)
    vv = Coarse_grain(v * v) - Coarse_grain(v) * Coarse_grain(v)
    ww = Coarse_grain(w * w) - Coarse_grain(w) * Coarse_grain(w)

    W = Coarse_grain(w)
    V = Coarse_grain(v)
    U = Coarse_grain(u)

    partu = Coarse_grain(w * u * u) - W * U * U - W * (Coarse_grain(u * u) - U * U) - 2 * U * (
                Coarse_grain(u * w) - U * W)
    partv = Coarse_grain(w * v * v) - W * V * V - W * (Coarse_grain(v * v) - V * V) - 2 * V * (
                Coarse_grain(v * w) - V * W)
    partw = Coarse_grain(w * w * w) - W * W * W - W * (Coarse_grain(w * w) - W * W) - 2 * W * (
                Coarse_grain(w * w) - W * W)
    we = partu + partv + partw

    return 0.5 * (ww + uu + vv).reshape(1, -1), ww.reshape(1, -1), 0.5 * we.reshape(1, -1)


def compute_ustar(u, v, w):
    w_s = (w[:1, :, :] + w[1:2, :, :]) / 2
    wv = Coarse_grain(w_s * v[:1, :, :]) - Coarse_grain(w_s) * Coarse_grain(v[:1, :, :])
    wu = Coarse_grain(w_s * u[:1, :, :]) - Coarse_grain(w_s) * Coarse_grain(u[:1, :, :])
    return np.power(wu ** 2 + wv ** 2, 0.25)


def bl_top(data):
    dt = np.gradient(np.gradient(np.mean(data, (1, 2))))
    return np.argmin(dt)


def make_filelist(path_file):
    searchfor = 'uins'
    fileType = '.out'
    direc = os.listdir(path_file)

    filelist = []

    for file in direc:
        if file.endswith(".out"):
            filelist.append(file)
    unique_files = []
    for ss in filelist:
        if 'uins' in ss:
            unique_files.append(ss[16:-4])
    return unique_files


def main():
    path_file = '/glade/scratch/sshamekh/cbl_'

    outfile = 'outputs_ny/'
    scale_flux = [0.01, 0.03, 0.06, 0.05, 0.03, 0.1, 0.1]
    names = ['16_01', '16_03', '16_06', '4_05', '8_03', '4_1', '2_1']
    for epoch in range(3):
        print(names[epoch])

        wstars = []
        ustars = []
        path = path_file + names[epoch] + '/output/'
        name = names[epoch]
        file_list = make_filelist(path)
        file_list = sorted(file_list, key=lambda x: int(x))
        print(len(file_list))
        for i in range(0, min(len(file_list), 98)):
            # for i in range(1):
            number = file_list[i]
            print(number, i)
            if name == '2_1':
                number2 = int(int(number) - int(number) % 500)
                if i == 0:
                    number2 += 500

            else:
                number2 = number[1:]

            #         # constructing the name of the outputfile
            fileinst_w = "aver_wins" + str(number) + str(number) + '.out'
            fileinst_u = "aver_uins" + str(number) + str(number) + '.out'
            fileinst_v = "aver_vins" + str(number) + str(number) + '.out'
            fileinst_theta = "aver_thetains" + str(number) + str(number) + '.out'

            fileinst_s1 = "aver_ps1ins" + str(number) + str(number) + '.out'
            fileinst_s2 = "aver_ps2ins" + str(number) + str(number) + '.out'
            fileinst_s3 = "aver_ps3ins" + str(number) + str(number) + '.out'
            # fileinst_p = "aver_pins"+str(number)+str(number)+'.out'

            filesgs_t = "aver_sgs_t30" + str(number2) + ".out"

            filesgs_p2sgs = "aver_sgspi20" + str(number2) + ".out"
            filesgs_p3sgs = "aver_sgspi30" + str(number2) + ".out"

            print("ins file is: ", fileinst_w)
            print('sgs file is: ', filesgs_t)

            #         print ("ins file is: ",fileinst_w, i )

            #         # reading data to numpy array

            w_velocity = (np.loadtxt(path + fileinst_w)).reshape(256, 256, 257)
            u_velocity = (np.loadtxt(path + fileinst_u)).reshape(256, 256, 257)
            v_velocity = (np.loadtxt(path + fileinst_v)).reshape(256, 256, 257)
            theta = (np.loadtxt(path + fileinst_theta)).reshape(256, 256, 257)

            s1_scalar = (np.loadtxt(path + fileinst_s1)).reshape(256, 256, 257)
            s2_scalar = (np.loadtxt(path + fileinst_s2)).reshape(256, 256, 257)
            s3_scalar = (np.loadtxt(path + fileinst_s3)).reshape(256, 256, 257)
            # p_scalar= (np.loadtxt(path + fileinst_p)).reshape(256,256,257)

            s2_sgs = (np.loadtxt(path + filesgs_p2sgs))  # .reshape(2*256,256,257)
            s3_sgs = (np.loadtxt(path + filesgs_p3sgs))  # .reshape(2*256,256,257)
            t_sgs = (np.loadtxt(path + filesgs_t))  # .reshape(2*256,256,257)
            print('shape:', t_sgs.shape)
            if t_sgs.shape[0] >= 70570:
                s2_sgs = s2_sgs.reshape(2 * 256, 256, 257)
                s3_sgs = s3_sgs.reshape(2 * 256, 256, 257)
                t_sgs = t_sgs.reshape(2 * 256, 256, 257)
                t_sgs = t_sgs[256:, :, 1:]
                s2_sgs = s2_sgs[256:, :, 1:]
                s3_sgs = s3_sgs[256:, :, 1:]
            else:
                s2_sgs = (s2_sgs).reshape(256, 256, 257)
                s3_sgs = (s3_sgs).reshape(256, 256, 257)
                t_sgs = (t_sgs).reshape(256, 256, 257)
                t_sgs = t_sgs[:, :, 1:]
                s2_sgs = s2_sgs[:, :, 1:]
                s3_sgs = s3_sgs[:, :, 1:]

            # #         # removing first layer. It contains time information
            pbl_h = bl_top(theta[1:200, :, :])
            ustar = compute_ustar(u_velocity[:, :, 1:], v_velocity[:, :, 1:], w_velocity[:, :, 1:])

            wstar = np.power((9.8 / 300) * scale_flux[epoch] * pbl_h * dz, 1 / 3.0)
            print('ustar: ', ustar, ' wstar: ', wstar, ' pnl h: ', pbl_h)
            w_velocity = w_velocity[:, :, 1:]
            u_velocity = shift(u_velocity[:, :, 1:])
            v_velocity = shift(v_velocity[:, :, 1:])
            theta = shift(theta[:, :, 1:])
            theta = theta * Tscale
            s1_scalar = shift(s1_scalar[:, :, 1:])
            s2_scalar = shift(s2_scalar[:, :, 1:])
            s3_scalar = shift(s3_scalar[:, :, 1:])
            # w_velocity = np.nan_to_num(w_velocity,nan = 0.0)

            tke, sigmaw, tkew = make_tke(w_velocity, u_velocity, v_velocity)

            all_mean = make_mean(theta, s1_scalar, s2_scalar, s3_scalar)
            all_95 = all_percentile(theta, s1_scalar, s2_scalar, s3_scalar, w_velocity, 95)

            flux = make_flux(theta, s1_scalar, s2_scalar, s3_scalar, w_velocity, t_sgs, s2_sgs, s3_sgs)

            theta3 = compute_third_moment(theta, theta, theta)
            theta2w = compute_third_moment(theta, theta, w_velocity)
            thetaw2 = compute_third_moment(theta, w_velocity, w_velocity)
            w3 = compute_third_moment(w_velocity, w_velocity, w_velocity)
            theta2 = compute_second_moment(theta, theta)
            if i == 0:
                scalar_mean = np.copy(all_mean)
                tkes = np.copy(tke)
                all_95s = np.copy(all_95)
                all_siw = np.copy(sigmaw)
                all_ew = np.copy(tkew)
                all_flux = np.copy(flux)
                all_theta3 = np.copy(theta3)
                all_theta2w = np.copy(theta2w)
                all_thetaw2 = np.copy(thetaw2)
                all_theta2 = np.copy(theta2)
                all_w3 = np.copy(w3)

            else:
                scalar_mean = np.concatenate((scalar_mean, all_mean), axis=0)
                tkes = np.concatenate((tkes, tke), axis=0)
                all_95s = np.concatenate((all_95s, all_95), axis=0)
                all_siw = np.concatenate((all_siw, sigmaw), axis=0)
                all_ew = np.concatenate((all_ew, tkew), axis=0)
                all_flux = np.concatenate((all_flux, flux), axis=0)
                all_theta3 = np.concatenate((all_theta3, theta3), axis=0)
                all_theta2w = np.concatenate((all_theta2w, theta2w), axis=0)
                all_thetaw2 = np.concatenate((all_thetaw2, thetaw2), axis=0)
                all_theta2 = np.concatenate((all_theta2, theta2), axis=0)
                all_w3 = np.concatenate((all_w3, w3), axis=0)

            wstars.append(wstar)
            ustars.append(ustar)
        np.save(outfile + name + '_v3' + '_scalar_mean', scalar_mean)
        np.save(outfile + name + '_v3' + '_tkes', tkes)
        np.save(outfile + name + '_v3' + '_percentiles', all_95s)
        np.save(outfile + name + '_v3' + '_sigmaw', all_siw)
        np.save(outfile + name + '_v3' + '_tkew', all_ew)
        np.save(outfile + name + '_v3' + '_wstarts', np.array(wstars))
        np.save(outfile + name + '_v3' + '_ustarts', np.array(ustars))
        np.save(outfile + name + '_v3' + '_fluxes', np.array(all_flux))

        np.save(outfile + name + '_v3' + '_theta3', np.array(all_theta3))
        np.save(outfile + name + '_v3' + '_theta2w', np.array(all_theta2w))
        np.save(outfile + name + '_v3' + '_thetaw2', np.array(all_thetaw2))
        np.save(outfile + name + '_v3' + '_theta2', np.array(all_theta2))
        np.save(outfile + name + '_v3' + '_w3', np.array(all_w3))


if __name__ == '__main__':
    main()
