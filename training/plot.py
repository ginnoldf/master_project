import matplotlib.pyplot as plt
import string
import numpy as np
from typing import Dict


def plot_means(true_mean: np.ndarray, true_std: np.ndarray, pred_mean: np.ndarray, filepath: string):
    plt.plot(range(len(true_mean)), true_mean, label='true mean')
    plt.fill_between(range(len(true_mean)), true_mean - true_std, true_mean + true_std, alpha=0.2)
    plt.plot(range(len(pred_mean)), pred_mean, label='predictions mean')
    plt.savefig(filepath)
    plt.clf()


def plot_samples(sample_evaluation: Dict, filepath: string):
    colors = ['b', 'g', 'r', 'c', 'm']
    for i, sample in enumerate(sample_evaluation):
        truth = sample['truth']
        pred = sample['pred'][0]
        plt.plot(range(len(truth)), truth, color=colors[i], linestyle='dashed')
        plt.plot(range(len(pred)), pred, color=colors[i], linestyle='solid')
    plt.savefig(filepath)
    plt.clf()
