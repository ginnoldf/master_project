import numpy as np


def main():

    sim_names = ['16_01', '16_03',  '16_06']
    for sim_name in sim_names:

        print('Verification of data for ' + sim_name + ':')

        # load saras data
        turb_heat_flux_sara = np.load('data/saras_data/' + sim_name + '_v3_fluxes.npy')[:, :, 0]
        theta_sara = np.load('data/saras_data/' + sim_name + '_v3_scalar_mean.npy')[:, :, 0]
        tkes_sara = np.load('data/saras_data/' + sim_name + '_v3_tkes.npy')
        num_timesteps = np.shape(turb_heat_flux_sara)[0]

        # load my data - cut last two elements, as they are not in saras vector
        turb_heat_flux = np.load('data/training_data/L_256/' + sim_name + '/turb_heat_flux.npy')[:num_timesteps]
        theta = np.load('data/training_data/L_256/' + sim_name + '/theta.npy')[:num_timesteps]
        tkes = np.load('data/training_data/L_256/' + sim_name + '/tkes.npy')[:num_timesteps]

        print(np.array_equal(turb_heat_flux[:, :, 0, 0], turb_heat_flux_sara))
        print(np.array_equal(theta[:, :, 0, 0], theta_sara))
        print(np.array_equal(tkes[:, :, 0, 0], tkes_sara))


if __name__ == '__main__':
    main()
