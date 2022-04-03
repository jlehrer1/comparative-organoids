import random
import sys
import argparse
import pathlib
import os
import ast
from typing import *

import comet_ml
import pandas as pd 
import torch
import numpy as np
import pytorch_lightning as pl
import wandb 

from functools import partial 
from torchmetrics.functional import accuracy, f1_score, precision, recall

from tqdm import tqdm 
import torch.nn as nn 
import torch.optim as optim
from pytorch_lightning.loggers import CometLogger, WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from helper import upload 
from lib.neural import GeneClassifier
from lib.data import generate_datasets, clean_sample

# Set all seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

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

def train_loop(model, trainloaders, valloaders, testloaders, refgenes):
    wandb.init()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(100):  # loop over the dataset multiple times
        print(f'On epoch {epoch}')
        running_loss = 0.0
        
        # Train loop
        for trainidx, trainloader in enumerate(trainloaders):
            model.train()
            print(f'Training on {trainidx}')
            
            for i, data in enumerate(tqdm(trainloader)):
                inputs, labels = data
                # CLEAN INPUTS
                inputs = clean_sample(inputs, refgenes, trainloader.dataset.dataset.columns)
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
                if i % 100 == 0:
                    running_loss = 0.0
                    metric_results = calculate_metrics(
                        outputs=outputs,
                        labels=labels,
                        append_str='train',
                        num_classes=model.N_labels
                    )
                    wandb.log({"train_loss": loss})
                    wandb.log(metric_results)
    
        # Validation loops 
        for validx, valloader in enumerate(valloaders):
            print(f'Evaluating on validation loader {validx}')
            model.eval()
            
            for i, data in enumerate(tqdm(valloader)):
                inputs, labels = data
                inputs = clean_sample(inputs, refgenes, valloader.dataset.dataset.columns)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                
                if i % 100 == 0:
                    running_loss = 0.0
                    metric_results = calculate_metrics(
                        outputs=outputs,
                        labels=labels,
                        append_str='val',
                        num_classes=model.N_labels
                    )
                    
                    wandb.log({"val_loss": loss})
                    wandb.log(metric_results)
        
    print('Finished train/validation, calculating test error')

    for testidx, testloader in enumerate(testloaders):
        print(f'Evaluating on test loader {testidx}')
        model.eval()
        
        for i, data in enumerate(tqdm(testloader)):
            inputs, labels = data
            inputs = clean_sample(inputs, refgenes, valloader.dataset.dataset.columns)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            if i % 100 == 0:
                running_loss = 0.0
                metric_results = calculate_metrics(
                    outputs=outputs,
                    labels=labels,
                    append_str='test',
                    num_classes=model.N_labels
                )

                wandb.log({"test_loss": loss})
                wandb.log(metric_results)

def calculate_metrics(
    outputs, 
    labels,
    num_classes,
    append_str='',
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
    
    for name, metric in metrics.items():
        res = metric(
            preds=outputs,
            target=labels,
        )
        
        results[f"{name}{f'_{append_str}' if append_str else ''}"] = res
    
    return results 