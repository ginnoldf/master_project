import argparse
import datetime
import os
import pathlib
import yaml
from typing import Dict, Mapping
import string
import torch
import collections.abc

import training.models as models


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file",
        type=pathlib.Path,
        default="runs/default.yaml",
        help="path to the file describing the data locations"
    )
    return parser


def update(default: Mapping, update_values: Mapping):
    result = default
    for key, value in update_values.items():
        if isinstance(value, collections.abc.Mapping):
            result[key] = update(default.get(key, {}), value)
        else:
            result[key] = update_values.get(key, {})
    return result


def get_model(model_dict: Dict):
    model = None
    if model_dict['name'] == 'simple':
        model = models.SimpleModel().double()
    if model_dict['name'] == 'dnn1':
        model = models.DNN1().double()
    if model_dict['name'] == 'cnn_dnn1':
        model = models.CNN_DNN1().double()
    if model_dict['name'] == 'cnn_dnn2':
        model = models.CNN_DNN2().double()

    # load model if specified
    if 'stateDictPath' in model_dict.keys() and not model_dict['stateDictPath'] is None:
        model.load_state_dict(torch.load(model_dict['stateDictPath']))

    # freeze model children
    for i, child in enumerate(model.children()):
        if 'freezeChildren' in model_dict.keys() and not model_dict['freezeChildren'] is None and i in model_dict['freezeChildren']:
            for param in child.parameters():
                param.requires_grad = False

    return model


def get_loss_fn(loss_name: string):
    if loss_name == 'mse':
        return torch.nn.MSELoss()
    else:
        return None


def get_optimizer(optimizer: Dict, model: torch.nn.Module):
    if optimizer['name'] == 'sgd':
        return torch.optim.SGD(model.parameters(),
                               lr=optimizer['lr'],
                               momentum=optimizer['momentum'],
                               weight_decay=optimizer['weightDecay'])
    if optimizer['name'] == 'adam':
        return torch.optim.Adam(model.parameters(),
                                lr=optimizer['lr'],
                                betas=eval(optimizer['betas']),
                                weight_decay=optimizer['weightDecay'])
    else:
        return None


def get_lr_scheduler(lr_scheduler: Dict, optimizer: torch.optim.Optimizer):
    if lr_scheduler['name'] == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                               step_size=lr_scheduler['stepStepSize'],
                                               gamma=lr_scheduler['stepGamma'])
    elif lr_scheduler['name'] == 'cosineAnnealingWarmRestarts':
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer,
                                                                    T_0=lr_scheduler['T0'],
                                                                    T_mult=lr_scheduler['TMult'],
                                                                    eta_min=lr_scheduler['etaMin'],
                                                                    last_epoch=lr_scheduler['lastEpoch'])
    else:
        return None


class TrainingConfig:
    def __init__(self):
        parser = build_argparser()

        # read default config file
        with open('training/default.yaml', 'r') as f:
            default_config = yaml.safe_load(f)

        # read real config file
        config_file_path = parser.parse_args().config_file
        with open(config_file_path, 'r') as f:
            self.config_dict = update(default_config, yaml.safe_load(f))

        # add timestamp
        self.config_dict['timestamp'] = datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

        self.logdir = self.config_dict['writer']['logdir']
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)

        # load device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # decide between maml and a normal optimizer run
        self.run = self.config_dict['run']
        if self.run == 'maml':
            self.bsize_base = self.config_dict['data']['batchSizeBase']
            self.base_datasets = self.config_dict['data']['baseDatasets']
            self.target_datasets = self.config_dict['data']['targetDatasets']
            self.lr_maml = self.config_dict['training']['mamlLR']

        if self.run == 'optimizer':
            self.train_datasets = self.config_dict['data']['trainDatasets']
            self.bsize = self.config_dict['data']['batchSize']

        # load data config
        self.data_config_path = self.config_dict['data']['dataConfigPath']
        self.train_split = self.config_dict['data']['trainSplit']

        self.epochs = self.config_dict['training']['epochs']
        self.eval_epochs = self.config_dict['training']['eval_epochs']
        self.model = get_model(self.config_dict['training']['model']).to(self.device)
        self.loss_fn = get_loss_fn(self.config_dict['training']['loss'])
        self.optimizer = get_optimizer(optimizer=self.config_dict['training']['optimizer'],
                                       model=self.model)
        self.lr_scheduler = get_lr_scheduler(lr_scheduler=self.config_dict['training']['lrScheduler'],
                                             optimizer=self.optimizer)

    def save(self, logdir):
        with open(os.path.join(logdir, 'config.yaml'), 'w') as f:
            yaml.dump(self.config_dict, f)
