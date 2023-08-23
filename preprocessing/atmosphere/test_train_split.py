import numpy as np
import yaml
import data_reshape
import os
import math


def main():
    # load data config
    with open('datasets/atmosphere/data_config.yaml', 'r') as file:
        data_config = yaml.safe_load(file)['datasets']

    # load all turbulent heat fluxes and calculate mean std in time axis
    for dataset in data_config:
        for directory in dataset['directories']:
            # load theta, tkes and turb_heat_flux
            theta = np.load(os.path.join(directory, 'theta.npy'))
            theta = data_reshape(theta)
            tkes = np.load(os.path.join(directory, 'tkes.npy'))
            tkes = data_reshape(tkes)
            turb_heat_flux = np.load(os.path.join(directory, 'turb_heat_flux.npy'))
            turb_heat_flux = data_reshape(turb_heat_flux)

            # create train and test directory
            train_dir = os.path.join(directory, 'train')
            if not os.path.exists(train_dir):
                os.makedirs(train_dir)

            test_dir = os.path.join(directory, 'test')
            if not os.path.exists(test_dir):
                os.makedirs(test_dir)

            # create split indexes
            dataset_length = len(theta)
            test_share = 0.2
            np.random.seed(0)
            indices = np.random.permutation(dataset_length)
            test_idx = indices[:math.floor(dataset_length*test_share)]
            train_idx = indices[math.floor(dataset_length*test_share):]

            # split data
            theta_test, theta_train = theta[test_idx], theta[train_idx]
            tkes_test, tkes_train = tkes[test_idx], tkes[train_idx]
            thf_test, thf_train = turb_heat_flux[test_idx], turb_heat_flux[train_idx]

            # save splitted data
            np.save(os.path.join(train_dir, "theta"), theta_train)
            np.save(os.path.join(test_dir, "theta"), theta_test)
            np.save(os.path.join(train_dir, "tkes"), tkes_train)
            np.save(os.path.join(test_dir, "tkes"), tkes_test)
            np.save(os.path.join(train_dir, "turb_heat_flux"), thf_train)
            np.save(os.path.join(test_dir, "turb_heat_flux"), thf_test)


if __name__ == '__main__':
    main()
