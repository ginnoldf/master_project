import numpy as np


def data_reshape(np_array: np.ndarray):
    (timesteps, z_dim, x_dim, y_dim) = np_array.shape
    num_training_samples = timesteps * x_dim * y_dim
    training_samples = np.zeros((num_training_samples, z_dim))
    for timestep in range(timesteps):
        for x_idx in range(x_dim):
            for y_idx in range(y_dim):
                training_samples[timestep * x_dim * y_dim + x_idx * y_dim + y_idx] = np_array[timestep, :, x_idx, y_idx]
    return training_samples
