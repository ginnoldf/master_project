import torch
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List

import learn2learn as l2l

from training.writer import Writer
from training.evaluation import evaluation, evaluate_dataset


def train(
        device: torch.device,
        writer: Writer,
        epochs: int,
        train_dataset_base: Dataset,
        train_dataset_target: Dataset,
        test_dataset_base: Dataset,
        test_dataset_target: Dataset,
        bsize_base: int,
        eval_dataloaders: List[Dict],
        optimizer: torch.optim.Optimizer,
        #lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        lr_scheduler,
        model: torch.nn.Module,
        lr_maml: float,
        loss_fn,
        eval_epochs: int
):
    # construct trainloaders for base and target tasks
    train_loader_base = DataLoader(train_dataset_base, batch_size=bsize_base, shuffle=True)
    train_loader_target = DataLoader(train_dataset_target, batch_size=len(train_dataset_target), shuffle=True)

    # create the maml model
    maml = l2l.algorithms.MAML(model, lr=lr_maml, first_order=False)

    for epoch in range(epochs):
        # train
        model.train()
        optimizer.zero_grad()

        # clone the model
        learner = maml.clone().double()

        # inner loop
        running_loss_inner = 0.0
        for i_inner, batch in enumerate(train_loader_base):
            # load batch data
            inputs, outputs_true = batch

            # make predictions for this batch
            outputs_model = learner(inputs.double().to(device))

            # Compute the loss
            loss = loss_fn(outputs_model.to(device), outputs_true.to(device))
            running_loss_inner += loss

            # adapt for base training set
            learner.adapt(loss)

        # normalize inner running loss
        running_loss_inner /= (i_inner + 1)

        # evaluation of inner loop on base test set
        test_loss_inner, _ = evaluate_dataset(device=device,
                                              model=learner,
                                              dataloader=DataLoader(test_dataset_base, batch_size=len(test_dataset_base), shuffle=False),
                                              loss_fn=loss_fn)

        # outer loop
        running_loss_outer = 0.0
        for i_outer, batch in enumerate(train_loader_target):
            # load batch data
            inputs, outputs_true = batch

            # make predictions for this batch
            outputs_model = learner(inputs.double().to(device))

            # compute the loss and its gradients
            loss = loss_fn(outputs_model.to(device), outputs_true.to(device))
            running_loss_outer += loss

        # normalize outer running loss
        running_loss_outer /= (i_outer + 1)

        # parameter outer loop update
        running_loss_outer.backward()
        optimizer.step()

        # lr scheduling
        lr_scheduler.step()

        # evaluation of inner loop on base test set
        test_loss_outer, _ = evaluate_dataset(device=device,
                                              model=learner,
                                              dataloader=DataLoader(test_dataset_target, batch_size=len(test_dataset_target), shuffle=False),
                                              loss_fn=loss_fn)

        # gather data and report
        writer.step_maml(global_step=epoch,
                         lr_opt=optimizer.state_dict()['param_groups'][0]['lr'],
                         loss_inner=running_loss_inner.item(),
                         loss_outer=running_loss_outer.item(),
                         test_loss_outer=test_loss_outer,
                         test_loss_inner=test_loss_inner)

        # test model
        if epoch % eval_epochs == 0:
            evaluation(device=device,
                       writer=writer,
                       epoch=epoch,
                       model=model,
                       eval_dataloaders=eval_dataloaders,
                       loss_fn=loss_fn,
                       global_step=epoch + 1)
