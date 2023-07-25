import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from typing import Dict, List

from training.writer import Writer


def evaluation(device: torch.device,
               writer: Writer,
               epoch: int,
               model: torch.nn.Module,
               eval_dataloaders: List[Dict],
               loss_fn,
               global_step: int):
    model.eval()
    with torch.no_grad():

        # evaluate model on all datasets
        all_datasets_evaluation = []
        for eval_dataloader in eval_dataloaders:
            # get avg loss for all datasets
            dataloader = eval_dataloader['dataloader']
            avg_loss, pred_mean = evaluate_dataset(device=device,
                                                   dataloader=dataloader,
                                                   model=model,
                                                   loss_fn=loss_fn)

            # get true turbulent heat flux distribution
            # special case of ConcatDatasets must be handled slightly differently
            if isinstance(dataloader.dataset, ConcatDataset):
                thf_array = []
                for dataset in dataloader.dataset.datasets:
                    thf_array.append(dataset.tensors[1].numpy())
                true_data = np.concatenate(thf_array, axis=0)
            else:
                true_data = dataloader.dataset[:][1].numpy()
            true_mean = np.mean(true_data, axis=0)
            true_std = np.std(true_data, axis=0)

            # draw random samples out of the dataset to compare prediction with truth
            sample_indices = np.random.randint(low=0, high=len(dataloader.dataset), size=3)
            sample_evaluation = []
            for sample_idx in sample_indices:
                truth = dataloader.dataset[sample_idx][1].numpy()
                pred = model(dataloader.dataset[sample_idx][0].to(device)).cpu().numpy()
                sample_evaluation.append({'truth': truth, 'pred': pred})

            all_datasets_evaluation.append({'dataset': eval_dataloader['dataset_name'],
                                            'avg_loss': avg_loss,
                                            'pred_mean': pred_mean,
                                            'true_mean': true_mean,
                                            'true_std': true_std,
                                            'sample_evaluation': sample_evaluation})

    writer.evaluation(global_step=global_step,
                      epoch=epoch,
                      all_datasets_evaluation=all_datasets_evaluation)


def evaluate_dataset(device: torch.device, model: torch.nn.Module, dataloader: DataLoader, loss_fn):
    with torch.no_grad():
        running_test_loss = 0.0
        running_pred_mean = np.zeros(len(dataloader.dataset[0][1]))
        for i, test_batch in enumerate(dataloader):
            test_inputs, test_outputs_true = test_batch
            test_outputs_model = model(test_inputs.double().to(device))
            test_loss = loss_fn(test_outputs_model.to(device), test_outputs_true.to(device))
            running_test_loss += test_loss

            # add up mean of predictions
            running_pred_mean += np.sum(test_outputs_model.cpu().numpy(), axis=0)

        avg_loss = running_test_loss.item() / (i + 1)
        pred_mean = running_pred_mean / len(dataloader.dataset)
    return avg_loss, pred_mean

