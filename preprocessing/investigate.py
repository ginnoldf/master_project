import numpy as np
import yaml
from training.data import data_reshape
import os
import matplotlib.pyplot as plt
import string

plt.rcParams.update({'font.size': 18})


# plot a distribution
def plot(mean: np.ndarray, std: np.ndarray, filepath: string):
    plt.plot(mean, range(len(mean)), label='mean')
    plt.fill_betweenx(range(len(mean)), mean - std, mean + std, alpha=0.2, label='std')
    plt.legend()
    plt.savefig(os.path.join(filepath))
    plt.clf()


# We use this script to investigate the strucure of the turbulent heat flux data that we want to predict. We want to
# ensure, that our models
def main():
    # load data config
    with open('datasets/training_data/data_config.yaml', 'r') as file:
        data_config = yaml.safe_load(file)['data']

    # load all turbulent heat fluxes and calculate mean std in time axis
    for lz_key in data_config:
        for lxy_key in data_config[lz_key]:
            print('investigate ' + lz_key + ', ' + lxy_key + ': ')
            for i, sim_key in enumerate(data_config[lz_key][lxy_key]):

                # load theta, tkes and turb_heat_flux
                theta = np.load(os.path.join(data_config[lz_key][lxy_key][sim_key], 'theta.npy'))
                theta = data_reshape(theta)
                tkes = np.load(os.path.join(data_config[lz_key][lxy_key][sim_key], 'tkes.npy'))
                tkes = data_reshape(tkes)
                turb_heat_flux = np.load(os.path.join(data_config[lz_key][lxy_key][sim_key], 'turb_heat_flux.npy'))
                turb_heat_flux = data_reshape(turb_heat_flux)

                # create plots to visualize the data
                plot(mean=np.mean(theta, axis=0),
                     std=np.std(theta, axis=0),
                     filepath=os.path.join(data_config[lz_key][lxy_key][sim_key], 'theta_plot.png'))

                plot(mean=np.mean(tkes, axis=0),
                     std=np.std(tkes, axis=0),
                     filepath=os.path.join(data_config[lz_key][lxy_key][sim_key], 'tkes_plot.png'))

                plot(mean=np.mean(turb_heat_flux, axis=0),
                     std=np.std(turb_heat_flux, axis=0),
                     filepath=os.path.join(data_config[lz_key][lxy_key][sim_key], 'turb_heat_flux_plot.png'))


if __name__ == '__main__':
    main()
