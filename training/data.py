import yaml
import numpy as np
import os
from typing import List

import torch
from torch.utils.data import TensorDataset, DataLoader, random_split, ConcatDataset

from training.config import TrainingConfig


# load data config to read file paths from
def load_datasets_config(data_config_path: str):
    with open(data_config_path, 'r') as file:
        data_config = yaml.safe_load(file)
    return data_config['datasets']


def data_reshape(np_array: np.ndarray):
    (timesteps, z_dim, x_dim, y_dim) = np_array.shape
    num_training_samples = timesteps * x_dim * y_dim
    training_samples = np.zeros((num_training_samples, z_dim))
    for timestep in range(timesteps):
        for x_idx in range(x_dim):
            for y_idx in range(y_dim):
                training_samples[timestep * x_dim * y_dim + x_idx * y_dim + y_idx] = np_array[timestep, :, x_idx, y_idx]
    return training_samples


def load_data(directories: List[str]):
    theta_np_arrays, tkes_np_arrays, turb_heat_flux_np_arrays = [], [], []

    # load all numpy arrays
    for directory in directories:
        theta_path = os.path.join(directory, 'theta.npy')
        tkes_path = os.path.join(directory, 'tkes.npy')
        turb_heat_flux_path = os.path.join(directory, 'turb_heat_flux.npy')
        theta_np_arrays.append(data_reshape(np.load(theta_path)))
        tkes_np_arrays.append(data_reshape(np.load(tkes_path)))
        turb_heat_flux_np_arrays.append(data_reshape(np.load(turb_heat_flux_path)))

    # concat numpy arrays to return a complete dataset
    theta = np.concatenate(theta_np_arrays)
    tke = np.concatenate(tkes_np_arrays)
    turb_heat_flux = np.concatenate(turb_heat_flux_np_arrays)

    return theta, tke, turb_heat_flux


def get_datasets(config: TrainingConfig):
    # read data from disk to numpy arrays
    datasets_config = load_datasets_config(config.data_config_path)

    # load all datasets that are described in the data config file
    eval_dataloaders = []
    train_datasets = []
    for dataset_config in datasets_config:
        # load data
        theta, tkes, turb_heat_flux = load_data(directories=dataset_config['directories'])

        # create structured input data
        in_np = np.stack((theta, tkes), axis=1)

        # create torch train and test datasets from numpy arrays
        dataset_size = theta.shape[0]
        dataset = TensorDataset(torch.from_numpy(in_np), torch.from_numpy(turb_heat_flux))

        # do a train test split on the dataset if we want to train on it
        if dataset_config['name'] in config.train_datasets:
            train_size = round(dataset_size * config.train_split)
            test_size = dataset_size - train_size
            train_set, test_set = random_split(dataset, [train_size, test_size])
            train_datasets.append(train_set)
            eval_dataloaders.append({'dataset_name': dataset_config['name'],
                                     'dataloader': DataLoader(test_set, batch_size=len(test_set), shuffle=False)})
        else:
            eval_dataloaders.append({'dataset_name': dataset_config['name'],
                                     'dataloader': DataLoader(dataset, len(dataset), shuffle=False)})

    return ConcatDataset(train_datasets), eval_dataloaders
