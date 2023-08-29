import yaml
import numpy as np
import os
from typing import List, Dict

import torch
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset

from training.config import TrainingConfig


def get_load_data_fn(data_category: str):
    if data_category == 'atmosphere':
        return load_data_atmosphere
    elif data_category == 'ocean':
        return load_data_ocean
    else:
        return None


# load data config to read file paths from
def load_datasets_config(data_config_path: str):
    with open(data_config_path, 'r') as file:
        data_config = yaml.safe_load(file)
    return data_config['datasets']


def load_ds_atmosphere(directories: List[str], subdir: str):
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


def load_data_atmosphere(dataset_config: Dict):
    # load data
    theta_train, tkes_train, thf_train = load_ds_atmosphere(directories=dataset_config['directories'], subdir='train')
    theta_test, tkes_test, thf_test = load_ds_atmosphere(directories=dataset_config['directories'], subdir='test')

    # create structured input data
    in_train = np.stack((theta_train, tkes_train), axis=1)
    in_test = np.stack((theta_test, tkes_test), axis=1)

    # create torch train and test datasets from numpy arrays
    dataset_train = TensorDataset(torch.from_numpy(in_train), torch.from_numpy(thf_train))
    dataset_test = TensorDataset(torch.from_numpy(in_test), torch.from_numpy(thf_test))

    return dataset_train, dataset_test


def load_data_ocean(dataset_config: Dict):
    # load data
    train_vars = {}
    test_vars = {}
    in_vars = ['U_x', 'U_y', 'V_x', 'V_y', 'Sx', 'Sy']
    out_vars = ['Sfnx', 'Sfny']
    for var in in_vars + out_vars:
        train_vars[var] = np.load(os.path.join(dataset_config['directory'], 'train', var + '.npy'))
        test_vars[var] = np.load(os.path.join(dataset_config['directory'], 'test', var + '.npy'))

    # create structured input and output data
    in_train = np.stack([train_vars[var] for var in in_vars], axis=1)
    in_test = np.stack([test_vars[var] for var in in_vars], axis=1)
    out_train = np.stack([train_vars[var] for var in out_vars], axis=1)
    out_test = np.stack([test_vars[var] for var in out_vars], axis=1)

    # create torch train and test datasets from numpy arrays
    dataset_train = TensorDataset(torch.from_numpy(in_train), torch.from_numpy(out_train))
    dataset_test = TensorDataset(torch.from_numpy(in_test), torch.from_numpy(out_test))

    return dataset_train, dataset_test


def get_data(config: TrainingConfig):
    if config.data_category == 'ocean' and config.run == 'optimizer':
        return get_data_opt(config, load_data_ocean)
    elif config.data_category == 'ocean' and config.run == 'maml':
        return get_data_maml(config, load_data_ocean)
    elif config.data_category == 'atmosphere' and config.run == 'optimizer':
        return get_data_opt(config, load_data_atmosphere)
    elif config.data_category == 'atmosphere' and config.run == 'optimizer':
        return get_data_maml(config, load_data_atmosphere())
    else:
        return None


def get_data_maml(config: TrainingConfig):
    # read data config
    datasets_config = load_datasets_config(config.data_config_path)

    # get data loading function
    load_data = get_load_data_fn(config.data_category)

    # load all datasets that are described in the data config file
    eval_dataloaders = []
    train_datasets_base = []
    train_datasets_target = []
    test_datasets_base = []
    test_datasets_target = []
    for dataset_config in datasets_config:
        # load data
        dataset_train, dataset_test = load_data(dataset_config)

        # handle target and base datasets
        if dataset_config['name'] in config.base_datasets and dataset_config['name'] in config.target_datasets:
            train_datasets_base.append(dataset_train)
            test_datasets_base.append(dataset_test)
            train_datasets_target.append(dataset_train)
            test_datasets_target.append(dataset_test)
            eval_dataloaders.append({'dataset_name': dataset_config['name'],
                                     'dataloader': DataLoader(dataset_test, batch_size=100000,
                                                              shuffle=False)})
        elif dataset_config['name'] in config.base_datasets:
            train_datasets_base.append(dataset_train)
            test_datasets_base.append(dataset_test)
            eval_dataloaders.append({'dataset_name': dataset_config['name'],
                                     'dataloader': DataLoader(dataset_test, batch_size=100000,
                                                              shuffle=False)})
        elif dataset_config['name'] in config.target_datasets:
            train_datasets_target.append(dataset_train)
            test_datasets_target.append(dataset_test)
            eval_dataloaders.append({'dataset_name': dataset_config['name'],
                                     'dataloader': DataLoader(dataset_test, batch_size=100000,
                                                              shuffle=False)})
        else:
            dataset_all = ConcatDataset([dataset_train, dataset_test])
            eval_dataloaders.append({'dataset_name': dataset_config['name'],
                                     'dataloader': DataLoader(dataset_all, batch_size=100000, shuffle=False)})

    return train_datasets_base, \
        train_datasets_target, \
        ConcatDataset(test_datasets_base), \
        ConcatDataset(test_datasets_target), \
        eval_dataloaders


def get_data_opt(config: TrainingConfig, data_category: str):
    # read data config
    datasets_config = load_datasets_config(config.data_config_path)

    # get data loading function
    load_data = get_load_data_fn(config.data_category)

    # load all datasets that are described in the data config file
    eval_dataloaders = []
    train_datasets = []
    for dataset_config in datasets_config:
        # load data
        print('load data for ' + dataset_config['name'])
        dataset_train, dataset_test = load_data(dataset_config)

        # dont evaluate on training data
        if dataset_config['name'] in config.train_datasets:
            train_datasets.append(dataset_train)
            eval_dataloaders.append({'dataset_name': dataset_config['name'],
                                     'dataloader': DataLoader(dataset_test, batch_size=100000, shuffle=False)})
        else:
            dataset_all = ConcatDataset([dataset_train, dataset_test])
            eval_dataloaders.append({'dataset_name': dataset_config['name'],
                                     'dataloader': DataLoader(dataset_all, batch_size=100000, shuffle=False)})

    return ConcatDataset(train_datasets), eval_dataloaders
