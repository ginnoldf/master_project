from config import TrainingConfig
from writer import Writer
import data
import train


def main():
    # load training configuration
    config = TrainingConfig()

    # create writer
    writer = Writer(config.logdir)

    # load data and structure it to train and test loader
    train_loader, test_loader = data.get_data_loaders(config=config)

    # training
    train.train(
        writer=writer,
        epochs=config.epochs,
        eval_epochs=config.eval_epochs,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=config.optimizer,
        lr_scheduler=config.lr_scheduler,
        model=config.model,
        loss_fn=config.loss_fn
    )


if __name__ == '__main__':
    main()
