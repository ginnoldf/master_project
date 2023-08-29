import torch

from training.config import TrainingConfig
from training.writer import Writer
import training.data as data
import training.train as train
import training.train_maml as train_maml


def main():
    # set seed for reproducability - especially for train and test splits
    torch.manual_seed(0)

    # load training configuration
    config = TrainingConfig()

    # create writer
    writer = Writer(config.logdir)

    # normal optimizer run
    if config.run == 'optimizer':
        # load data and structure it to train, test and eval datasets
        print('start loading data')
        train_dataset, eval_dataloaders = data.get_data(config=config)
        print('data loaded')

        # training
        train.train(
            device=config.device,
            writer=writer,
            epochs=config.epochs,
            eval_epochs=config.eval_epochs,
            train_dataset=train_dataset,
            bsize=config.bsize,
            eval_dataloaders=eval_dataloaders,
            optimizer=config.optimizer,
            lr_scheduler=config.lr_scheduler,
            model=config.model,
            loss_fn=config.loss_fn,
            data_category=config.data_category,
            plotting=config.plotting
        )

    # maml run
    if config.run == 'maml':
        # load data and structure it to train, test and eval datasets for base and target
        train_datasets_base, train_datasets_target, test_dataset_base, test_dataset_target, eval_dataloaders = data.get_data_maml(config)
        print('data loaded')

        # training
        train_maml.train(
            device=config.device,
            writer=writer,
            epochs=config.epochs,
            eval_epochs=config.eval_epochs,
            maml_k=config.maml_k,
            train_datasets_base=train_datasets_base,
            train_datasets_target=train_datasets_target,
            test_dataset_base=test_dataset_base,
            test_dataset_target=test_dataset_target,
            bsize_base=config.bsize_base,
            eval_dataloaders=eval_dataloaders,
            optimizer=config.optimizer,
            lr_scheduler=config.lr_scheduler,
            model=config.model,
            lr_maml=config.lr_maml,
            loss_fn=config.loss_fn,
            data_category=config.data_category,
            plotting=config.plotting
        )

    writer.end(config=config)


if __name__ == '__main__':
    main()
