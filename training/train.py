import torch
from torch.utils.data import DataLoader
from typing import Dict

from writer import Writer
from evaluation import evaluation


def train_one_epoch(
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

        # Make predictions for this batch
        outputs_model = model(inputs.double())

        # Compute the loss and its gradients
        loss = loss_fn(outputs_model, outputs_true)
        loss.backward()

        # actual learning step
        optimizer.step()

        # Gather data and report
        writer.step(global_step=global_steps_start + (i + 1) * train_loader.batch_size,
                    lr=optimizer.state_dict()['param_groups'][0]['lr'],
                    loss_per_sample=loss.item())


def train(
        writer: Writer,
        epochs: int,
        train_loader: DataLoader,
        test_loader: DataLoader,
        all_dataloaders: Dict,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        model: torch.nn.Module,
        loss_fn,
        eval_epochs: int
):
    for epoch in range(epochs):
        # train
        model.train()
        train_one_epoch(writer=writer,
                        train_loader=train_loader,
                        model=model,
                        loss_fn=loss_fn,
                        optimizer=optimizer,
                        global_steps_start=epoch * len(train_loader.dataset))

        # lr scheduling
        lr_scheduler.step()

        # test model
        if epoch % eval_epochs == 0:
            evaluation(writer=writer,
                       epoch=epoch,
                       model=model,
                       test_loader=test_loader,
                       all_dataloaders=all_dataloaders,
                       loss_fn=loss_fn,
                       global_step=len(train_loader.dataset) * (epoch + 1))
