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


# plot for one directory
def plot_for_dir(directory, reshape=False):
    # load theta, tkes and turb_heat_flux
    theta = np.load(os.path.join(directory, 'theta.npy'))
    tkes = np.load(os.path.join(directory, 'tkes.npy'))
    turb_heat_flux = np.load(os.path.join(directory, 'turb_heat_flux.npy'))

    if reshape:
        theta = data_reshape(theta)
        tkes = data_reshape(tkes)
        turb_heat_flux = data_reshape(turb_heat_flux)

    # create plots to visualize the data
    plot(mean=np.mean(theta, axis=0),
         std=np.std(theta, axis=0),
         filepath=os.path.join(directory, 'theta_plot.png'))

    plot(mean=np.mean(tkes, axis=0),
         std=np.std(tkes, axis=0),
         filepath=os.path.join(directory, 'tkes_plot.png'))

    plot(mean=np.mean(turb_heat_flux, axis=0),
         std=np.std(turb_heat_flux, axis=0),
         filepath=os.path.join(directory, 'turb_heat_flux_plot.png'))


# We use this script to investigate the structure of the turbulent heat flux data that we want to predict. We want to
# ensure, that our models
def main():
    # load data config
    with open('datasets/data_config.yaml', 'r') as file:
        data_config = yaml.safe_load(file)['datasets']

    # load all turbulent heat fluxes and calculate mean std in time axis
    for dataset in data_config:
        for directory in dataset['directories']:
            print('investigate ' + directory)
            # plots for whole dataset
            plot_for_dir(directory, reshape=True)

            # plots for test set
            plot_for_dir(os.path.join(directory, 'test'))

            # plots for train set
            plot_for_dir(os.path.join(directory, 'train'))


if __name__ == '__main__':
    main()
