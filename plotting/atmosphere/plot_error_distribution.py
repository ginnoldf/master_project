import torch
import numpy as np
import os
import matplotlib.pyplot as plt

import training.models as models


def main():
    dataset_path = 'datasets/lz2/lxy64/16_06/test'

    # load models
    model_1 = models.CNN_DNN2().double()
    model_2 = models.CNN_DNN2().double()
    model_all = models.CNN_DNN2().double()
    model_1.load_state_dict(torch.load('runs/maml/cnn_dnn2_run6/01/model.pt', map_location='cpu'))
    model_2.load_state_dict(torch.load('runs/maml/cnn_dnn2_run6/06/model.pt', map_location='cpu'))
    model_all.load_state_dict(torch.load('runs/cnn_dnn2/cosine_annealing/all/model.pt', map_location='cpu'))

    # load data
    theta = np.load(os.path.join(dataset_path, 'theta.npy'))
    tkes = np.load(os.path.join(dataset_path, 'tkes.npy'))
    thf = np.load(os.path.join(dataset_path, 'turb_heat_flux.npy'))

    # get thf predictions
    in_array = np.stack((theta, tkes), axis=1)
    pred_1 = model_1(torch.from_numpy(in_array)).detach().numpy()
    pred_2 = model_2(torch.from_numpy(in_array)).detach().numpy()
    pred_all = model_all(torch.from_numpy(in_array)).detach().numpy()

    # calculate errors
    errors_1 = np.mean(np.square(thf - pred_1), axis=1)
    errors_2 = np.mean(np.square(thf - pred_2), axis=1)
    errors_all = np.mean(np.square(thf - pred_all), axis=1)

    # plot error distribution
    plt.hist([errors_1, errors_2, errors_all],
             bins=np.arange(0, 0.06, 0.004),
             #alpha=0.8,
             label=['MAML 1', 'MAML 2', 'model trained on all datasets'],
             color=['tab:blue', 'tab:red',  'tab:orange'])
    #plt.xlim(0, 2)
    plt.xlabel('mean squared error')
    plt.ylabel('number of samples')
    plt.legend()
    plt.savefig('post_run_evaluation/plots/error_distributions/error_distributions_maml_64_06.png')
    plt.clf()

    # plot some samples
    for sample in range(20):
        plt.plot(thf[sample], range(len(thf[sample])), linestyle='dashed', label='truth', color='tab:green')
        plt.plot(pred_1[sample], range(len(pred_1[sample])), linestyle='solid', label='prediction MAML 1', color='tab:blue')
        plt.plot(pred_2[sample], range(len(pred_2[sample])), linestyle='solid', label='prediction MAML 2', color='tab:red')
        plt.plot(pred_all[sample], range(len(pred_all[sample])), linestyle='solid', label='prediction model all', color='tab:orange')
        plt.xlabel('turbulent heat flux')
        plt.ylabel('z')
        plt.legend()
        plt.savefig('post_run_evaluation/plots/samples/maml_64_06/sample_' + str(sample) + '.png')
        plt.clf()

    return


if __name__ == '__main__':
    main()
