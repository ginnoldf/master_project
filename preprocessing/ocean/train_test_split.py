import numpy as np
import yaml
import os
import math


def main():
    # load data config
    with open('datasets/ocean/data_config.yaml', 'r') as file:
        data_config = yaml.safe_load(file)['datasets']

    # load all variables
    variables = ['Sfnx', 'Sfny', 'U_x', 'U_y', 'V_x', 'V_y', 'Sx', 'Sy']
    for dataset in data_config:
        vars_dict = {}
        for var in variables:
            vars_dict[var] = np.load(os.path.join(dataset['directory'], var + '.npy')).flatten()

        # create split indexes
        dataset_length = len(vars_dict[variables[0]])
        test_share = 0.2
        np.random.seed(0)
        indices = np.random.permutation(dataset_length)
        test_idx = indices[:math.floor(dataset_length*test_share)]
        train_idx = indices[math.floor(dataset_length*test_share):]

        # create train and test directory if they do not exist
        train_dir = os.path.join(dataset['directory'], 'train')
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)

        test_dir = os.path.join(dataset['directory'], 'test')
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        # split and save data
        for var in variables:
            np.save(os.path.join(train_dir, var), vars_dict[var][train_idx])
            np.save(os.path.join(test_dir, var), vars_dict[var][test_idx])


if __name__ == '__main__':
    main()
