import string
import os
import numpy as np

import torch
from tensorboardX import SummaryWriter

from training.config import TrainingConfig
from training.plot import plot_means, plot_samples


class Writer:
    def __init__(self, logdir: string):
        self.logdir = logdir
        self.numpy_dir = os.path.join(self.logdir, 'numpy')
        if not os.path.exists(self.numpy_dir):
            os.makedirs(self.numpy_dir)
        self.tb_writer = SummaryWriter(logdir=logdir)
        self.steps = np.array([])
        self.eval_dict = {}

    def step(self, global_step, lr, loss_per_batch):
        #self.tb_writer.add_scalar('learning rate', scalar_value=lr, global_step=global_step)
        #self.tb_writer.add_scalar('loss per train batch', scalar_value=loss_per_batch, global_step=global_step)
        return

    def step_maml(self, global_step, lr_opt, loss_inner, loss_outer, test_loss_outer, test_loss_inner):
        #self.tb_writer.add_scalar('learning rate', scalar_value=lr_opt, global_step=global_step)
        self.tb_writer.add_scalar('train loss inner loop', scalar_value=loss_inner, global_step=global_step)
        self.tb_writer.add_scalar('train loss outer loop', scalar_value=loss_outer, global_step=global_step)
        #self.tb_writer.add_scalar('test loss outer loop', scalar_value=test_loss_outer, global_step=global_step)
        #self.tb_writer.add_scalar('test loss inner loop', scalar_value=test_loss_inner, global_step=global_step)

    def epoch(self, global_step, avg_loss):
        self.tb_writer.add_scalar('avg loss over epoch', scalar_value=avg_loss, global_step=global_step)

    def evaluation(self, model, global_step, epoch, all_datasets_evaluation, data_category, plotting):
        # save model
        torch.save(model.state_dict(), os.path.join(self.logdir, 'model.pt'))

        # save step number for later plotting
        self.steps = np.append(self.steps, [global_step])
        np.save(os.path.join(self.numpy_dir, 'steps'), self.steps)

        # write evaluation on all datasets
        for dataset_evaluation in all_datasets_evaluation:
            self.tb_writer.add_scalar(dataset_evaluation['dataset'],
                                      scalar_value=dataset_evaluation['avg_loss'],
                                      global_step=global_step)

            # save dataset evaluation numpy files
            if not dataset_evaluation['dataset'] in self.eval_dict:
                self.eval_dict[dataset_evaluation['dataset']] = np.array([])
            self.eval_dict[dataset_evaluation['dataset']] = np.append(self.eval_dict[dataset_evaluation['dataset']],
                                                                      [dataset_evaluation['avg_loss']])
            np.save(os.path.join(self.numpy_dir, dataset_evaluation['dataset']), self.eval_dict[dataset_evaluation['dataset']])

            # create plots for prediction mean and samples
            if data_category == 'atmosphere' and plotting is True:
                # create plot directory if necessary and plot means
                means_plot_dir = os.path.join(self.logdir, 'eval', dataset_evaluation['dataset'], 'means')
                if not os.path.exists(means_plot_dir):
                    os.makedirs(means_plot_dir)
                means_plot_filepath = os.path.join(means_plot_dir, 'epoch' + str(epoch) + '.png')
                plot_means(true_mean=dataset_evaluation['true_mean'],
                           true_std=dataset_evaluation['true_std'],
                           pred_mean=dataset_evaluation['pred_mean'],
                           filepath=means_plot_filepath)

                # create plot directory if necessary and plot samples
                samples_plot_dir = os.path.join(self.logdir, 'eval', dataset_evaluation['dataset'], 'samples')
                if not os.path.exists(samples_plot_dir):
                    os.makedirs(samples_plot_dir)
                samples_plot_filepath = os.path.join(samples_plot_dir, 'epoch' + str(epoch) + '.png')
                plot_samples(sample_evaluation=dataset_evaluation['sample_evaluation'],
                             filepath=samples_plot_filepath)

    def end(self, config: TrainingConfig):
        # save config of the experiment
        config.save(logdir=self.logdir)
