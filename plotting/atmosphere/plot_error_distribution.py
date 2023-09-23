import torch
import numpy as np
import os
import matplotlib.pyplot as plt

import training.models

MODEL_PATHS = {
    'MAML1': 'runs/atmosphere/maml/from_pretrained/lz2_lxy16/02/model.pt',
    'MAML2': 'runs/atmosphere/maml/randomly_initialized/02/model.pt',
    'model trained on all datasets': 'runs/atmosphere/opt/step_lr/all/model.pt',
    'model trained on Lxy=16': 'runs/atmosphere/opt/step_lr/lz2_lxy16/model.pt',
    'model trained on Lxy=32': 'runs/atmosphere/opt/step_lr/lz2_lxy32/model.pt',
    'model trained on Lxy=64': 'runs/atmosphere/opt/step_lr/lz2_lxy64/model.pt',
    'model trained on Lxy=128': 'runs/atmosphere/opt/step_lr/lz2_lxy128/model.pt',
    'model trained on Lxy=256': 'runs/atmosphere/opt/step_lr/lz2_lxy256/model.pt',
}

DATASET_PATHS = {
    'lz2_lxy64_01': 'datasets/atmosphere/lz2/lxy64/16_01/test',
    'lz2_lxy64_03': 'datasets/atmosphere/lz2/lxy64/16_03/test',
    'lz2_lxy64_06': 'datasets/atmosphere/lz2/lxy64/16_06/test',
    'lz2_lxy16_01': 'datasets/atmosphere/lz2/lxy16/16_01/test',
    'lz2_lxy16_03': 'datasets/atmosphere/lz2/lxy16/16_03/test',
    'lz2_lxy16_06': 'datasets/atmosphere/lz2/lxy16/16_06/test',
}


def plot_error_dist_and_samples(dataset_name,
                                model_names,
                                plot_name,
                                bins=None,
                                color=None):
    # load data
    theta = np.load(os.path.join(DATASET_PATHS[dataset_name], 'theta.npy'))
    tkes = np.load(os.path.join(DATASET_PATHS[dataset_name], 'tkes.npy'))
    thf = np.load(os.path.join(DATASET_PATHS[dataset_name], 'turb_heat_flux.npy'))
    in_array = np.stack((theta, tkes), axis=1)

    # calculate errors and predictions for all models
    models = {}
    predictions = {}
    errors = {}
    for model_name in model_names:
        # load model
        models[model_name] = training.models.AtmosphereModel().double()
        models[model_name].load_state_dict(torch.load(MODEL_PATHS[model_name], map_location='cpu'))

        # predict
        predictions[model_name] = models[model_name](torch.from_numpy(in_array)).detach().numpy()

        # calculate error
        errors[model_name] = np.mean(np.square(thf - predictions[model_name]), axis=1)

    # plot error distribution
    plt.hist([errors[model_name] for model_name in model_names],
             bins=bins,
             alpha=0.7,
             label=[model_name for model_name in model_names],
             color=color)
    # plt.xlim(0, 2)
    plt.xlabel('mean squared error')
    plt.ylabel('number of samples')
    plt.legend()
    plt.savefig(os.path.join('plotting/plots/error_distributions', plot_name + '.png'))
    plt.clf()

    # plot some samples
    for sample in range(30):
        plt.plot(thf[sample], range(len(thf[sample])), linestyle='dashed', label='truth', color='tab:green')
        for i, model_name in enumerate(predictions):
            plt.plot(predictions[model_name][sample],
                     range(len(predictions[model_name][sample])),
                     linestyle='solid',
                     label=model_name,
                     color=color[i])
        plt.xlabel('vertical heat flux')
        plt.ylabel('z')
        plt.legend()

        # create directory
        directory = os.path.join('plotting/plots/samples/', plot_name)
        if not os.path.exists(directory):
            os.makedirs(directory)

        plt.savefig(os.path.join(directory, str(sample) + '.png'))
        plt.clf()


def main():

    # MAML plots
    plot_error_dist_and_samples(dataset_name='lz2_lxy64_01',
                                model_names=['MAML1', 'MAML2', 'model trained on all datasets'],
                                plot_name='maml_64_01',
                                bins=np.arange(0, 0.1, 0.005),
                                color=['tab:blue', 'tab:red', 'tab:orange'])
    plot_error_dist_and_samples(dataset_name='lz2_lxy64_03',
                                model_names=['MAML1', 'MAML2', 'model trained on all datasets'],
                                plot_name='maml_64_03',
                                bins=np.arange(0, 0.1, 0.005),
                                color=['tab:blue', 'tab:red', 'tab:orange'])
    plot_error_dist_and_samples(dataset_name='lz2_lxy64_06',
                                model_names=['MAML1', 'MAML2', 'model trained on all datasets'],
                                plot_name='maml_64_06',
                                bins=np.arange(0, 0.1, 0.005),
                                color=['tab:blue', 'tab:red', 'tab:orange'])

    plot_error_dist_and_samples(dataset_name='lz2_lxy16_01',
                                model_names=['MAML1', 'MAML2', 'model trained on all datasets'],
                                plot_name='maml_16_01',
                                bins=np.arange(0, 2, 0.1),
                                color=['tab:blue', 'tab:red', 'tab:orange'])
    plot_error_dist_and_samples(dataset_name='lz2_lxy16_03',
                                model_names=['MAML1', 'MAML2', 'model trained on all datasets'],
                                plot_name='maml_16_03',
                                bins=np.arange(0, 2, 0.1),
                                color=['tab:blue', 'tab:red', 'tab:orange'])
    plot_error_dist_and_samples(dataset_name='lz2_lxy16_06',
                                model_names=['MAML1', 'MAML2', 'model trained on all datasets'],
                                plot_name='maml_16_06',
                                bins=np.arange(0, 2, 0.1),
                                color=['tab:blue', 'tab:red', 'tab:orange'])

    # baseline models
    plot_error_dist_and_samples(dataset_name='lz2_lxy64_01',
                                model_names=['model trained on Lxy=64', 'model trained on all datasets'],
                                plot_name='baseline_64_01',
                                bins=np.arange(0, 0.1, 0.005),
                                color=['tab:blue', 'tab:orange'])
    plot_error_dist_and_samples(dataset_name='lz2_lxy64_03',
                                model_names=['model trained on Lxy=64', 'model trained on all datasets'],
                                plot_name='baseline_64_03',
                                bins=np.arange(0, 0.1, 0.005),
                                color=['tab:blue', 'tab:orange'])
    plot_error_dist_and_samples(dataset_name='lz2_lxy64_06',
                                model_names=['model trained on Lxy=64', 'model trained on all datasets'],
                                plot_name='baseline_64_06',
                                bins=np.arange(0, 0.1, 0.005),
                                color=['tab:blue', 'tab:orange'])

    plot_error_dist_and_samples(dataset_name='lz2_lxy16_01',
                                model_names=['model trained on Lxy=16', 'model trained on all datasets'],
                                plot_name='baseline_16_01',
                                bins=np.arange(0, 2, 0.1),
                                color=['tab:blue', 'tab:orange'])
    plot_error_dist_and_samples(dataset_name='lz2_lxy16_03',
                                model_names=['model trained on Lxy=16', 'model trained on all datasets'],
                                plot_name='baseline_16_03',
                                bins=np.arange(0, 2, 0.1),
                                color=['tab:blue', 'tab:orange'])
    plot_error_dist_and_samples(dataset_name='lz2_lxy16_06',
                                model_names=['model trained on Lxy=16', 'model trained on all datasets'],
                                plot_name='baseline_16_06',
                                bins=np.arange(0, 2, 0.1),
                                color=['tab:blue', 'tab:orange'])



if __name__ == '__main__':
    main()
