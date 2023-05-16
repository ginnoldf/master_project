import torch
from torch.utils.data import DataLoader
from typing import Dict

from writer import Writer


def evaluation(writer: Writer,
               epoch: int,
               model: torch.nn.Module,
               test_loader: DataLoader,
               all_dataloaders: Dict,
               loss_fn,
               global_step: int):
    model.eval()
    with torch.no_grad():
        # evaluate model on test data
        avg_test_loss = evaluate_dataset(dataloader=test_loader, model=model, loss_fn=loss_fn)

        # evaluate model on all datasets
        all_datasets_evaluation = []
        for lz_key in all_dataloaders:
            for lxy_dataloader in all_dataloaders[lz_key]:
                avg_loss = evaluate_dataset(dataloader=lxy_dataloader['dataloader'], model=model, loss_fn=loss_fn)
                all_datasets_evaluation.append({'metric': lz_key + ' ' + lxy_dataloader['lxy_key'] + ' avg loss',
                                                'avg_loss': avg_loss})

    writer.evaluation(global_step=global_step,
                      epoch=epoch,
                      avg_test_loss=avg_test_loss,
                      all_datasets_evaluation=all_datasets_evaluation)


def evaluate_dataset(model: torch.nn.Module, dataloader: DataLoader, loss_fn):
    running_test_loss = 0.0
    for i, test_batch in enumerate(dataloader):
        test_inputs, test_outputs_true = test_batch
        test_outputs_model = model(test_inputs.double())
        test_loss = loss_fn(test_outputs_model, test_outputs_true)
        running_test_loss += test_loss
    avg_loss = running_test_loss.item() / len(dataloader.dataset)
    return avg_loss
