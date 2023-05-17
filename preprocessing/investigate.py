import numpy as np
import yaml
from training.data import data_reshape


# We use this script to investigate the strucure of the turbulent heat flux data that we want to predict. We want to
# ensure, that our models
def main():
    # load data config
    with open('datasets/training_data/data_config.yaml', 'r') as file:
        data_config = yaml.safe_load(file)['data']

    # load all turbulent heat fluxes and calculate mean std in time axis
    for lz_key in data_config:
        for lxy_key in data_config[lz_key]:
            print('investigate ' + lz_key + ', ' + lxy_key + ': ')
            all_sim_turb_heat_fluxes = []
            for i, sim_key in enumerate(data_config[lz_key][lxy_key]):
                # calculate and print all mean standard deviations
                turb_heat_flux = np.load(data_config[lz_key][lxy_key][sim_key]['turbHeatFlux'])
                all_sim_turb_heat_fluxes.append(data_reshape(turb_heat_flux))

            all_sim_turb_heat_fluxes_np = np.concatenate(all_sim_turb_heat_fluxes)
            mean_std_data = np.mean(np.std(all_sim_turb_heat_fluxes_np, axis=0))
            print('mean std over time, x and y: ' + str(mean_std_data))

            # calculate MSE to average for each height
            means = np.mean(all_sim_turb_heat_fluxes_np, axis=0)
            mean_array = means.reshape(1, len(means))\
                .repeat(all_sim_turb_heat_fluxes_np.shape[0], axis=0)
            mse = np.mean(((all_sim_turb_heat_fluxes_np - mean_array) ** 2))
            print('mse with mean over time, x and y: ' + str(mse))


if __name__ == '__main__':
    main()
