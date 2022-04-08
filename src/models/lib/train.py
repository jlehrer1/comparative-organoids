import random
import sys
import os
from typing import *

import pandas as pd 
import torch
import numpy as np
import pytorch_lightning as pl
import wandb 

from functools import partial 
from torchmetrics.functional import accuracy, f1_score, precision, recall
from sklearn.utils.class_weight import compute_class_weight

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from helper import upload, seed_everything
from .data import clean_sample

# Set all seeds for reproducibility
seed_everything(42)

class UploadCallback(pl.callbacks.Callback):
    """Custom PyTorch callback for uploading model checkpoints to the braingeneers S3 bucket.
    
    Parameters:
    path: Local path to folder where model checkpoints are saved
    desc: Description of checkpoint that is appended to checkpoint file name on save
    upload_path: Subpath in braingeneersdev/jlehrer/ to upload model checkpoints to
    """
    
    def __init__(self, path, desc, upload_path='model_checkpoints') -> None:
        super().__init__()
        self.path = path 
        self.desc = desc
        self.upload_path = upload_path

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % 10 == 0: # Save every ten epochs
            checkpoint = f'checkpoint-{epoch}-desc-{self.desc}.ckpt'
            trainer.save_checkpoint(os.path.join(self.path, checkpoint))
            print(f'Uploading checkpoint at epoch {epoch}')
            upload(
                os.path.join(self.path, checkpoint),
                os.path.join('jlehrer', self.upload_path, checkpoint)
            )

# reproducibility over all workers
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def train_val_loop(
    model, 
    trainloaders, 
    valloaders, 
    refgenes,
    criterion,
    optimizer,
    mod,
):
    wandb.init()
    wandb.watch(model)
    for epoch in range(1000):  # loop over the dataset multiple times
        print(f'{epoch = }')
        running_loss = 0.0
        epoch_loss = 0.0
        # Train loop
        model.train()
        for train in trainloaders:
            for i, data in enumerate(train):
                inputs, labels = data
                # CLEAN INPUTS
                inputs = clean_sample(inputs, refgenes, train.dataset.columns)
                # Forward pass ➡
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Backward pass ⬅
                optimizer.zero_grad()
                loss.backward()

                # Step with optimizer
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                epoch_loss += loss.item()


                if i % mod == 0: # record every 2000 mini batches 
                    metric_results = calculate_metrics(
                        outputs=outputs,
                        labels=labels,
                        append_str='train',
                        num_classes=model.output_dim,
                        subset='weighted_accuracy',
                    )

                    wandb.log(metric_results)
                    running_loss = running_loss / mod
                    wandb.log({f"batch_train_loss": loss})

                    running_loss = 0.0
                
        wandb.log({f"epoch_train_loss": epoch_loss / len(train)})
        
        model.eval()
        with torch.no_grad(): # save memory but not computing gradients 
            running_loss = 0.0
            epoch_loss = 0.0
            
            for val in valloaders:
                for i, data in enumerate(val):
                    inputs, labels = data
                    # CLEAN INPUTS
                    inputs = clean_sample(inputs, refgenes, val.dataset.columns)
                    # Forward pass ➡
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # print statistics
                    running_loss += loss.item()
                    epoch_loss += loss.item()

                    if i % mod == 0: #every 2000 mini batches 
                        running_loss = running_loss / mod
                        wandb.log({"val_loss": loss})
                        running_loss = 0.0

                        metric_results = calculate_metrics(
                            outputs=outputs,
                            labels=labels,
                            num_classes=model.output_dim,
                            subset='weighted_accuracy',
                            append_str='val',
                        )

                    wandb.log(metric_results)
        
            wandb.log({f"epoch_val_loss": epoch_loss / len(train)})

def test_loop(
    model,
    testloaders,
    refgenes,
    criterion,
    mod,
):
    model.eval()
    
    with torch.no_grad():
        for test in testloaders:
            running_loss = 0.0
            for i, data in enumerate(test):
                inputs, labels = data
                # CLEAN INPUTS
                inputs = clean_sample(inputs, refgenes, test.dataset.columns)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # print statistics
                running_loss += loss.item()
                if i % mod == 0: #every 2000 mini batches 
                    running_loss = running_loss / mod
                    wandb.log({"test_loss": loss})
                    running_loss = 0.0

                    metric_results = calculate_metrics(
                        outputs=outputs,
                        labels=labels,
                        num_classes=model.output_dim,
                        subset='weighted_accuracy',
                        append_str='test',
                    )

                    wandb.log(metric_results)

def calculate_metrics(
    outputs: torch.Tensor, 
    labels: torch.Tensor,
    num_classes: int,
    append_str: str='',
    subset: List[str]=None,
) -> Dict[str, float]:

    metrics = {
        'micro_accuracy': partial(accuracy, average='micro', num_classes=num_classes),
        'macro_accuracy': partial(accuracy, average='macro', num_classes=num_classes),
        'weighted_accuracy': partial(accuracy, average='weighted', num_classes=num_classes),
        'f1': f1_score,
        'precision': precision,
        'recall': recall,
    }
    results = {}

    subset = ([subset] if isinstance(subset, str) else subset)
    subset = (metrics.keys() if subset is None else subset)

    for name in subset:
        metric = metrics[name]
        res = metric(
            preds=outputs,
            target=labels,
        )
        
        results[f"{name}{f'_{append_str}' if append_str else ''}"] = res
    
    return results 

def _inner_computation(
    data,
    model, 
    optimizer,
    criterion,
    i, 
    running_loss,
    refgenes,
    currgenes,
    mode=['train', 'val', 'test'],
    wandb=None,
    record=None,
    quiet=True,
):
    inputs, labels = data
    inputs = clean_sample(inputs, refgenes, currgenes)
    outputs, M_loss = model(inputs)
    loss = criterion(outputs, labels)
    
    if mode == 'train':
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    running_loss += loss.item()

    if i % 100 == 0:
        running_loss = running_loss / 100
        metric_results = calculate_metrics(
            outputs=outputs,
            labels=labels,
            append_str=mode,
            num_classes=model.output_dim
        )
        
        if record is not None:
            record.append(running_loss)
            
        if wandb is not None:
            wandb.log({f"{mode}_loss": loss})
            wandb.log(metric_results)
            
        if not quiet:
            print(metric_results)
        print(f'{mode} loss is {running_loss}')
            
        running_loss = 0.0
    
    return running_loss, record

def total_class_weights(
    labelfiles: List[str],
    class_label: str,
) -> torch.Tensor:
    """
    Compute class weights for the entire label set of N labels.

    Parameters:
    labelfiles: List of absolute paths to label files

    Returns:
    np.array: Array of class weights for classes 0,...,N-1
    """

    comb = []

    for file in labelfiles:
        comb.extend(
            pd.read_csv(file).loc[:, class_label].values
        )

    return torch.from_numpy(compute_class_weight(
        classes=np.unique(comb),
        y=comb,
        class_weight='balanced',
    )).float()

def num_labels(labelfiles, class_label):
    val = []
    for file in labelfiles:
        val.append(pd.read_csv(file).loc[:, class_label].values.max())
    return max(val)