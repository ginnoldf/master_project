import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from typing import Dict, List

from training.writer import Writer
from training.evaluation import evaluation


def train_one_epoch(
        device: torch.device,
        writer: Writer,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        model: torch.nn.Module,
        loss_fn,
        global_steps_start: int
):
    for i, batch in enumerate(train_loader):
        # load batch data
        inputs, outputs_true = batch

        # zero your gradients for every batch
        optimizer.zero_grad()

        # make predictions for this batch
        outputs_model = model(inputs.double().to(device))

        # compute the loss and its gradients
        loss = loss_fn(outputs_model.to(device), outputs_true.to(device))
        loss.backward()

        # actual learning step
        optimizer.step()

        # gather data and report
        writer.step(global_step=global_steps_start + (i + 1) * train_loader.batch_size,
                    lr=optimizer.state_dict()['param_groups'][0]['lr'],
                    loss_per_batch=loss.item())


def train(
        device: torch.device,
        writer: Writer,
        epochs: int,
        train_dataset: Dataset,
        bsize: int,
        eval_dataloaders: List[Dict],
        optimizer: torch.optim.Optimizer,
        #lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        lr_scheduler,
        model: torch.nn.Module,
        loss_fn,
        eval_epochs: int
):

    for epoch in range(epochs):
        # train
        model.train()
        train_one_epoch(device=device,
                        writer=writer,
                        train_loader=DataLoader(train_dataset, batch_size=bsize, shuffle=True),
                        model=model,
                        loss_fn=loss_fn,
                        optimizer=optimizer,
                        global_steps_start=epoch * len(train_dataset))

        # lr scheduling
        lr_scheduler.step()

        # test model
        if epoch % eval_epochs == 0:
            evaluation(device=device,
                       writer=writer,
                       epoch=epoch,
                       model=model,
                       eval_dataloaders=eval_dataloaders,
                       loss_fn=loss_fn,
                       global_step=len(train_dataset) * (epoch + 1))
