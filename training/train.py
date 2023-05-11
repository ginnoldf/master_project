import torch
from torch.utils.data import DataLoader


def train_one_epoch(
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        model: torch.nn.Module,
        loss_fn
):
    running_loss = 0.
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
        running_loss += loss.item()
        if i % 20 == 19:
            last_loss = running_loss / 1000
            print('batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.


def train(
        epochs: int,
        train_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        model: torch.nn.Module,
        loss_fn
):
    for epoch in range(epochs):
        print('epoch {}:'.format(epoch))

        # train
        model.train()
        train_one_epoch(
            train_loader=train_loader,
            optimizer=optimizer,
            model=model,
            loss_fn=loss_fn
        )

        # test model
        if epoch % 100 == 99:
            model.eval()
            running_test_loss = 0.0
            for i, test_batch in enumerate(test_loader):
                test_inputs, test_outputs_true = test_batch
                test_outputs_model = model(test_inputs.double())
                test_loss = loss_fn(test_outputs_model, test_outputs_true)
                running_test_loss += test_loss
            avg_test_loss = running_test_loss / (i + 1)
            print('test loss {}'.format(avg_test_loss))

