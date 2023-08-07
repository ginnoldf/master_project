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


def load_data(directories: List[str], subdir: str):
    theta_np_arrays, tkes_np_arrays, thf_np_arrays = [], [], []

    # load all numpy arrays
    for directory in directories:
        theta_path = os.path.join(directory, subdir, 'theta.npy')
        tkes_path = os.path.join(directory, subdir, 'tkes.npy')
        thf_path = os.path.join(directory, subdir, 'turb_heat_flux.npy')
        theta_np_arrays.append(np.load(theta_path))
        tkes_np_arrays.append(np.load(tkes_path))
        thf_np_arrays.append(np.load(thf_path))

    # concat numpy arrays to return a complete dataset
    theta = np.concatenate(theta_np_arrays)
    tkes = np.concatenate(tkes_np_arrays)
    thf = np.concatenate(thf_np_arrays)

    return theta, tkes, thf


def get_data(config: TrainingConfig):
    # read data from disk to numpy arrays
    datasets_config = load_datasets_config(config.data_config_path)

    # load all datasets that are described in the data config file
    eval_dataloaders = []
    train_datasets = []
    for dataset_config in datasets_config:
        # load data
        theta_train, tkes_train, thf_train = load_data(directories=dataset_config['directories'], subdir='train')
        theta_test, tkes_test, thf_test = load_data(directories=dataset_config['directories'], subdir='test')

        # create structured input data
        in_train = np.stack((theta_train, tkes_train), axis=1)
        in_test = np.stack((theta_test, tkes_test), axis=1)

        # create torch train and test datasets from numpy arrays
        dataset_train = TensorDataset(torch.from_numpy(in_train), torch.from_numpy(thf_train))
        dataset_test = TensorDataset(torch.from_numpy(in_test), torch.from_numpy(thf_test))

        # do a train test split on the dataset if we want to train on it
        if dataset_config['name'] in config.train_datasets:
            train_datasets.append(dataset_train)
            eval_dataloaders.append({'dataset_name': dataset_config['name'],
                                     'dataloader': DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=False)})
        else:
            dataset_all = ConcatDataset([dataset_train, dataset_test])
            eval_dataloaders.append({'dataset_name': dataset_config['name'],
                                     'dataloader': DataLoader(dataset_all, batch_size=len(dataset_all), shuffle=False)})

    return ConcatDataset(train_datasets), eval_dataloaders


def get_data_maml(config: TrainingConfig):
    # read data from disk to numpy arrays
    datasets_config = load_datasets_config(config.data_config_path)

    # load all datasets that are described in the data config file
    eval_dataloaders = []
    train_datasets_base = []
    train_datasets_target = []
    test_datasets_base = []
    test_datasets_target = []
    for dataset_config in datasets_config:
        # load data
        theta_train, tkes_train, thf_train = load_data(directories=dataset_config['directories'], subdir='train')
        theta_test, tkes_test, thf_test = load_data(directories=dataset_config['directories'], subdir='test')

        # create structured input data
        in_train = np.stack((theta_train, tkes_train), axis=1)
        in_test = np.stack((theta_test, tkes_test), axis=1)

        # create torch train and test datasets from numpy arrays
        dataset_train = TensorDataset(torch.from_numpy(in_train), torch.from_numpy(thf_train))
        dataset_test = TensorDataset(torch.from_numpy(in_test), torch.from_numpy(thf_test))

        # do a train test split on the dataset if we want to train on it
        if dataset_config['name'] in config.base_datasets and dataset_config['name'] in config.target_datasets:
            train_datasets_base.append(dataset_train)
            test_datasets_base.append(dataset_test)
            train_datasets_target.append(dataset_train)
            test_datasets_target.append(dataset_test)
            eval_dataloaders.append({'dataset_name': dataset_config['name'],
                                     'dataloader': DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=False)})
        elif dataset_config['name'] in config.base_datasets:
            train_datasets_base.append(dataset_train)
            test_datasets_base.append(dataset_test)
            eval_dataloaders.append({'dataset_name': dataset_config['name'],
                                     'dataloader': DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=False)})
        elif dataset_config['name'] in config.target_datasets:
            train_datasets_target.append(dataset_train)
            test_datasets_target.append(dataset_test)
            eval_dataloaders.append({'dataset_name': dataset_config['name'],
                                     'dataloader': DataLoader(dataset_test, batch_size=len(dataset_test), shuffle=False)})
        else:
            dataset_all = ConcatDataset([dataset_train, dataset_test])
            eval_dataloaders.append({'dataset_name': dataset_config['name'],
                                     'dataloader': DataLoader(dataset_all, batch_size=len(dataset_all), shuffle=False)})

    return train_datasets_base, \
        train_datasets_target, \
        ConcatDataset(test_datasets_base), \
        ConcatDataset(test_datasets_target), \
        eval_dataloaders
