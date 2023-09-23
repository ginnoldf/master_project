import matplotlib.pyplot as plt
import os


def plot_table(validation_data, mse_dict, file_name):
    # create plot dir
    plot_dir = os.path.join('plotting', 'plots', 'mses')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # plot mse dict
    for dataset in mse_dict:
        plt.plot(validation_data, mse_dict[dataset], marker='o', label=dataset)
    plt.yscale('log')
    plt.xlabel('validation data')
    plt.ylabel('MSE')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, file_name))
    plt.clf()


def main():

    validation_data = ['Lxy=16', 'Lxy=32', 'Lxy=64', 'Lxy=128', 'Lxy=256']

    atm_baseline_mses = {
        'trained on Lxy=16': [0.092, 0.23, 0.7, 0.69, 0.62],
        'trained on Lxy=32': [0.19, 0.029, 0.093, 0.11, 0.088],
        'trained on Lxy=64': [0.19, 0.067, 0.0046, 0.016, 0.0017],
        'trained on Lxy=128': [0.24, 0.096, 0.026, 0.0006, 0.0014],
        'trained on Lxy=256': [0.23, 0.12, 0.052, 0.011, 0.000068],
        'trained on all datasets': [0.097, 0.039, 0.011, 0.0033, 0.0012],
    }
    plot_table(validation_data=validation_data, mse_dict=atm_baseline_mses, file_name='atm_baseline.png')

    atm_maml_mses = {
        'trained on corresponding dataset': [0.092, .029, 0.0046, 0.0006, 0.00007],
        'trained on all datasets': [0.097, 0.039, 0.011, 0.0033, 0.0012],
        'trained using MAML1': [0.16, 0.024, 0.0052, 0.0008, 0.0001],
        'trained using MAML2': [0.1, 0.03, 0.008, 0.001, 0.00018]
    }
    plot_table(validation_data=validation_data, mse_dict=atm_maml_mses, file_name='atm_maml.png')

    atm_maml_targeted_mses = {
        'trained on corresponding dataset': [0.092, .029, 0.0046, 0.0006, 0.00007],
        'MAML1 targeted Lxy=16': [0.09, 0.05, 0.04, 0.05, 0.03],
        'MAML1 targeted Lxy=32': [0.15, 0.025, 0.04, 0.07, 0.08],
        'MAML1 targeted Lxy=64': [0.18, 0.06, 0.004, 0.01, 0.01],
        'MAML1 targeted Lxy=128': [0.18, 0.07, 0.02, 0.0005, 0.0009],
        'MAML1 targeted Lxy=256': [0.18, 0.07, 0.02, 0.005, 0.00005],
    }
    plot_table(validation_data=validation_data, mse_dict=atm_maml_targeted_mses, file_name='atm_maml_targeted.png')

    atm_maml_targeted_reduced_mses = {
        'trained on corresponding dataset': [0.092, .029, 0.0046, 0.0006, 0.00007],
        'MAML1 targeted corresponding dataset': [0.09, 0.025, 0.004, 0.0005, 0.00005],
    }
    plot_table(validation_data=validation_data, mse_dict=atm_maml_targeted_reduced_mses, file_name='atm_maml_targeted_reduced.png')


if __name__ == '__main__':
    main()
