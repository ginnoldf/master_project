import torch
import os
import numpy as np
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from training.models import AtmosphereModel
from training.data import load_datasets_config

MODEL_PATHS = {
    #'maml1': 'runs/atmosphere/maml/from_pretrained/lz2_lxy16/02/model.pt',
    #'maml2': 'runs/atmosphere/maml/randomly_initialized/02/model.pt',
    'trained_on_all': 'runs/atmosphere/opt/step_lr/all/model.pt',
    'trained_on_lxy16': 'runs/atmosphere/opt/step_lr/lz2_lxy16/model.pt',
    'trained_on_lxy32': 'runs/atmosphere/opt/step_lr/lz2_lxy32/model.pt',
    'trained_on_lxy64': 'runs/atmosphere/opt/step_lr/lz2_lxy64/model.pt',
    'trained_on_lxy128': 'runs/atmosphere/opt/step_lr/lz2_lxy128/model.pt',
    'trained_on_lxy256': 'runs/atmosphere/opt/step_lr/lz2_lxy256/model.pt',
}

DATASET_NAMES = {
    'all': ['lz2_lxy256', 'lz2_lxy128', 'lz2_lxy64', 'lz2_lxy32', 'lz2_lxy16'],
    'lz2_lxy256': ['lz2_lxy256'],
    'lz2_lxy128': ['lz2_lxy128'],
    'lz2_lxy64': ['lz2_lxy64'],
    'lz2_lxy32': ['lz2_lxy32'],
    'lz2_lxy16': ['lz2_lxy16'],
}


def get_attr_means(model_name, dataset_name, data_config_path='datasets/atmosphere/data_config.yaml', num_samples=100):
    # load model
    model = AtmosphereModel()
    model.load_state_dict(torch.load(MODEL_PATHS[model_name], map_location='cpu'))

    # load data
    datasets_config = load_datasets_config(data_config_path)
    theta = []
    tkes = []
    thf = []
    for dataset_config in datasets_config:
        if dataset_config['name'] in DATASET_NAMES[dataset_name]:
            for directory in dataset_config['directories']:
                theta.append(np.load(os.path.join(directory, 'test', 'theta.npy')))
                tkes.append(np.load(os.path.join(directory, 'test', 'tkes.npy')))
                thf.append(np.load(os.path.join(directory, 'test', 'turb_heat_flux.npy')))

    in_tensor = torch.from_numpy(np.stack((np.concatenate(theta), np.concatenate(tkes)), axis=1)).double()
    out_tensor = torch.from_numpy(np.concatenate(thf)).double()

    # only consider random samples due to runtime
    random_indices = np.random.permutation(len(in_tensor))
    indices = random_indices[:num_samples]
    in_tensor = in_tensor[indices]
    out_tensor = out_tensor[indices]

    # apply integrated gradients
    ig = IntegratedGradients(model.double())
    in_tensor.requires_grad_()
    attr_mean = np.empty(90, dtype=object)
    for out_layer in range(90):
        attr = ig.attribute(in_tensor, target=out_layer, return_convergence_delta=False)
        attr_mean[out_layer] = np.mean(attr.detach().numpy(), axis=0)

    attr_mean_theta = np.transpose(np.stack([x[0] for x in attr_mean], axis=1))
    attr_mean_tke = np.transpose(np.stack([x[1] for x in attr_mean], axis=1))

    return attr_mean_theta, attr_mean_tke


def get_attr_relation(attr_means_base, attr_means, epsilon=0.01):
    # take the absolute of all attributions
    attr_means_base = np.absolute(attr_means_base)
    attr_means = np.absolute(attr_means)

    # mask values close to zero in all attributions
    attr_means_base[attr_means_base < epsilon] = epsilon
    attr_means[attr_means < epsilon] = epsilon

    # get quotient of attributions
    attr_relation = np.divide(attr_means, attr_means_base)
    return attr_relation


def plot_heat_map(attr_means, norm, directory, file_name):
    # create directory
    if not os.path.exists(directory):
        os.makedirs(directory)

    # create plot
    plt.imshow(attr_means, cmap='bwr', norm=norm, interpolation='nearest', origin='lower')
    plt.colorbar()
    plt.ylabel("output layer")
    plt.xlabel("input layer")
    plt.savefig(os.path.join(directory, file_name))
    plt.clf()
    print('created plot ' + os.path.join(directory, file_name))


def all_heat_map_plots():
    for model_name in MODEL_PATHS:
        for dataset_name in DATASET_NAMES:
            attr_means_theta, attr_means_tke = get_attr_means(model_name, dataset_name)
            directory = os.path.join('exp_ai/heat_maps_ig', model_name, dataset_name)
            plot_heat_map(attr_means=attr_means_theta,
                          norm=TwoSlopeNorm(vcenter=0, vmin=-0.2, vmax=0.2),
                          directory=directory,
                          file_name='theta.png')
            plot_heat_map(attr_means=attr_means_tke,
                          norm=TwoSlopeNorm(vcenter=0, vmin=-0.3, vmax=0.3),
                          directory=directory,
                          file_name='tke.png')


def plot_relation_curves(attr_relation_means, directory, file_name):
    # create directory
    if not os.path.exists(directory):
        os.makedirs(directory)

    for dataset_name in attr_relation_means:
        plt.plot(attr_relation_means[dataset_name], label=dataset_name)
    plt.xlabel("input layer")
    plt.ylabel("relative importance")
    plt.ylim(0.7, 4)
    plt.legend()
    plt.savefig(os.path.join(directory, file_name))
    plt.clf()
    print('created plot ' + os.path.join(directory, file_name))


def relation_plots_for_model(model_name, dataset_base_name, directory, cutoff=70):
    attr_means_base_theta, _ = get_attr_means(model_name, dataset_base_name)
    attr_relation_means = {}
    for dataset_name in DATASET_NAMES:
        if not dataset_name == dataset_base_name:
            # calculate attribution relation
            attr_means_theta, _ = get_attr_means(model_name, dataset_name)
            attr_relation = get_attr_relation(attr_means_base_theta, attr_means_theta)

            # plot
            plot_heat_map(attr_means=attr_relation,
                          norm=TwoSlopeNorm(vcenter=1, vmin=0, vmax=10),
                          directory=directory,
                          file_name=dataset_name + '.png')

            # reduce dimension to general input layer importance
            attr_relation_means[dataset_name] = np.mean(attr_relation[:][:cutoff], axis=1)

    # plot all relation curves for one model
    plot_relation_curves(attr_relation_means=attr_relation_means,
                         directory=directory,
                         file_name='relation_curves.png')


def all_relation_plots():
    directory = os.path.join('exp_ai/heat_maps_ig/relations')
    dataset_base_name = 'lz2_lxy16'
    for model_name in MODEL_PATHS:
        relation_plots_for_model(model_name=model_name,
                                 dataset_base_name=dataset_base_name,
                                 directory=os.path.join(directory, model_name))


def main():
    np.random.seed(0)
    all_heat_map_plots()
    #all_relation_plots()


if __name__ == '__main__':
    main()
