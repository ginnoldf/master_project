from writer import Writer

import torch
from torch.utils.data import DataLoader


def evaluation(writer: Writer,
               epoch: int,
               model: torch.nn.Module,
               test_loader: DataLoader,
               loss_fn,
               global_step: int):
    model.eval()
    running_test_loss = 0.0
    for i, test_batch in enumerate(test_loader):
        test_inputs, test_outputs_true = test_batch
        test_outputs_model = model(test_inputs.double())
        test_loss = loss_fn(test_outputs_model, test_outputs_true)
        running_test_loss += test_loss
    avg_test_loss = running_test_loss.item() / len(test_loader.dataset)
    writer.evaluation(global_step=global_step, epoch=epoch, avg_test_loss=avg_test_loss)
