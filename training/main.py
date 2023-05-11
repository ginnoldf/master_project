import torch

from utils import TrainingConfig
import training.data as data
from models import SimpleModel
import train


def main():
    # load training configuration
    config = TrainingConfig()

    # load data and structure it to train and test loader
    train_loader, test_loader = data.get_data_loaders(config)

    # define model, loss and optimizer
    model = SimpleModel().double()
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)

    # training
    train.train(
        epochs=config.epochs,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        model=model,
        loss_fn=loss_fn
    )


if __name__ == '__main__':
    main()
