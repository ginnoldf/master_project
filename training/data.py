import string
import yaml
from typing import Dict
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

from training.config import TrainingConfig


# load data config to read file paths from
def load_data_config(data_config_path: string):
    with open(data_config_path, 'r') as file:
        data_config = yaml.safe_load(file)
    return data_config['data']


def data_reshape(np_array: np.ndarray):
    (timesteps, z_dim, x_dim, y_dim) = np_array.shape
    num_training_samples = timesteps * x_dim * y_dim
    training_samples = np.zeros((num_training_samples, z_dim))
    for timestep in range(timesteps):
        for x_idx in range(x_dim):
            for y_idx in range(y_dim):
                training_samples[timestep * x_dim * y_dim + x_idx * y_dim + y_idx] = np_array[timestep, :, x_idx, y_idx]
    return training_samples


def load_data(data_config: Dict, lz_key: string, lxy_key: string):
    theta_np_arrays, tkes_np_arrays, turb_heat_flux_np_arrays = [], [], []

    # load all numpy arrays
    for sim in data_config[lz_key][lxy_key]:
        theta_np_arrays.append(data_reshape(np.load(data_config[lz_key][lxy_key][sim]['theta'])))
        tkes_np_arrays.append(data_reshape(np.load(data_config[lz_key][lxy_key][sim]['tke'])))
        turb_heat_flux_np_arrays.append(data_reshape(np.load(data_config[lz_key][lxy_key][sim]['turbHeatFlux'])))

    # concat numpy arrays to return a complete dataset
    theta = np.concatenate(theta_np_arrays)
    tke = np.concatenate(tkes_np_arrays)
    turb_heat_flux = np.concatenate(turb_heat_flux_np_arrays)

    return theta, tke, turb_heat_flux


def get_data_loaders(config: TrainingConfig):
    # read data from disk to numpy arrays
    data_config = load_data_config(config.data_config_path)

    # load all datasets that are described in the data config file
    all_dataloaders = {}
    train_loader, test_loader = (None, None)
    for lz_key in data_config:
        all_dataloaders[lz_key] = []
        for lxy_key in data_config[lz_key]:
            # load data
            theta, tkes, turb_heat_flux = load_data(data_config=data_config, lz_key=lz_key, lxy_key=lxy_key)

            # create structured input data
            in_np = np.stack((theta, tkes), axis=1)

            # create torch train and test datasets from numpy arrays
            dataset_size = theta.shape[0]
            dataset = TensorDataset(torch.from_numpy(in_np), torch.from_numpy(turb_heat_flux))

            # the set that we use to train, needs to be split into a train and a test set
            if lz_key == 'lz' + str(config.lz) and lxy_key == 'lxy' + str(config.lxy):
                train_size = round(dataset_size * config.train_split)
                test_size = dataset_size - train_size
                training_set, test_set = random_split(dataset, [train_size, test_size])

                # create torch dataloader from datasets
                train_loader = DataLoader(training_set, batch_size=config.bsize, shuffle=True)
                test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

            all_dataloaders[lz_key].append({'lxy_key': lxy_key,
                                            'dataloader': DataLoader(dataset, batch_size=32, shuffle=False)})

    return train_loader, test_loader, all_dataloaders
