import argparse
import pathlib
import yaml
from typing import Dict
import string
import torch

import models


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file",
        type=pathlib.Path,
        default="training/runs/default.yaml",
        help="path to the file describing the data locations"
    )
    return parser


def get_model(model: Dict):
    if model['name'] == 'simple':
        return models.SimpleModel().double()
    else:
        return None


def get_loss_fn(loss_name: string):
    if loss_name == 'mse':
        return torch.nn.MSELoss()
    else:
        return None


def get_optimizer(optimizer: Dict, model: torch.nn.Module):
    if optimizer['name'] == 'sgd':
        return torch.optim.SGD(model.parameters(),
                               lr=optimizer['lr'],
                               momentum=optimizer['momentum'])
    else:
        return None


def get_lr_scheduler(lr_scheduler: Dict, optimizer: torch.optim.Optimizer):
    if lr_scheduler['name'] == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                               step_size=lr_scheduler['stepStepSize'],
                                               gamma=lr_scheduler['stepGamma'])
    else:
        return None


class TrainingConfig:
    def __init__(self):
        parser = build_argparser()

        # read config file
        config_file_path = parser.parse_args().config_file
        with open(config_file_path) as f:
            config_dict = yaml.safe_load(f)

        self.logdir = config_dict['writer']['logdir']

        # load data config
        self.data_config_path = config_dict['data']['dataConfigPath']
        self.lz = config_dict['data']['lz']
        self.lxy = config_dict['data']['lxy']
        self.bsize = config_dict['data']['batchSize']
        self.train_split = config_dict['data']['trainSplit']

        self.epochs = config_dict['training']['epochs']
        self.eval_epochs = config_dict['training']['eval_epochs']
        self.model = get_model(config_dict['training']['model'])
        self.loss_fn = get_loss_fn(config_dict['training']['loss'])
        self.optimizer = get_optimizer(optimizer=config_dict['training']['optimizer'],
                                       model=self.model)
        self.lr_scheduler = get_lr_scheduler(lr_scheduler=config_dict['training']['lrScheduler'],
                                             optimizer=self.optimizer)
