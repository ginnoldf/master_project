from training.config import TrainingConfig
from training.writer import Writer
import training.data as data
import training.train as train


def main():
    # load training configuration
    config = TrainingConfig()

    # create writer
    writer = Writer(config.logdir)

    # load data and structure it to train, test and eval datasets
    train_dataset, eval_dataloaders = data.get_datasets(config=config)

    # training
    train.train(
        writer=writer,
        epochs=config.epochs,
        eval_epochs=config.eval_epochs,
        train_dataset=train_dataset,
        bsize=config.bsize,
        eval_dataloaders=eval_dataloaders,
        optimizer=config.optimizer,
        lr_scheduler=config.lr_scheduler,
        model=config.model,
        loss_fn=config.loss_fn
    )

    writer.end(config=config)


if __name__ == '__main__':
    main()
