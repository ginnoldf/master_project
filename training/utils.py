import argparse
import pathlib
import yaml


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file",
        type=pathlib.Path,
        default="training/runs/default.yaml",
        help="path to the file describing the data locations"
    )
    return parser


class TrainingConfig:
    def __init__(self):
        parser = build_argparser()

        # read config file
        config_file_path = parser.parse_args().config_file
        with open(config_file_path) as f:
            config_dict = yaml.safe_load(f)

        # load data config
        self.data_config_path = config_dict['data']['dataConfigPath']
        self.lz = config_dict['data']['lz']
        self.lxy = config_dict['data']['lxy']
        self.bsize = config_dict['data']['batchSize']
        self.train_split = config_dict['data']['trainSplit']

        self.epochs = config_dict['training']['epochs']
        self.model
        self.lr = args.lr
        self.momentum = args.momentum
