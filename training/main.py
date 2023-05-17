from training.config import TrainingConfig
from training.writer import Writer
import training.data as data
import training.train as train


def main():
    # load training configuration
    config = TrainingConfig()

    # create writer
    writer = Writer(config.logdir)

    # load data and structure it to train and test loader
    train_loader, test_loader, all_dataloaders = data.get_data_loaders(config=config)

    # training
    train.train(
        writer=writer,
        epochs=config.epochs,
        eval_epochs=config.eval_epochs,
        train_loader=train_loader,
        test_loader=test_loader,
        all_dataloaders=all_dataloaders,
        optimizer=config.optimizer,
        lr_scheduler=config.lr_scheduler,
        model=config.model,
        loss_fn=config.loss_fn
    )

    writer.end(config=config)


if __name__ == '__main__':
    main()
