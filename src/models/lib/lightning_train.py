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

from tqdm import tqdm 
import torch.nn as nn 
import torch.optim as optim
from pytorch_lightning.loggers import CometLogger, WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader

import sys, os 
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from neural import GeneClassifier
from train import UploadCallback, seed_worker
from data import generate_datasets

from helper import seed_everything

# Set all seeds for reproducibility
seed_everything(42)

def generate_trainer(
    here: str, 
    params: Dict[str, float], 
    class_label: str,
    num_workers: int,
    batch_size: int,
    weighted_metrics: bool,
    datafiles: List[str],
    labelfiles: List[str],
):
    """
    Generates PyTorch Lightning trainer and datasets for model training.

    Parameters:
    here: Absolute path to __file__
    params: Dictionary of hyperparameters for model training

    Returns:
    Tuple[trainer, model, traindata, valdata]: Tuple of PyTorch-Lightning trainer, model instance, and train and validation dataloaders for training.
    """

    device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device is {device}')

    width = params['width']
    epochs = params['epochs']
    layers = params['layers']

    data_path = os.path.join(here, '..', '..', 'data', 'processed')

    wandb_logger = WandbLogger(
        project=f"cell-classifier-{class_label}",
        name=f'{layers + 5} Layers, {width} Width'
    )

    train, test, input_size, num_labels, class_weights = generate_datasets(
        dataset_files=datafiles, # TODO: add this list of files as a parameter that can be passed to the training script, test this for now 
        label_files=labelfiles,
        class_label=class_label,
    )

    class_weights = class_weights.to(device)
    g = torch.Generator()
    g.manual_seed(42)

    traindata = DataLoader(
        train,
        batch_size=batch_size, 
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
    )

    valdata = DataLoader(
        test, 
        batch_size=batch_size, 
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
    )

    uploadcallback = UploadCallback(
        path=os.path.join(here, 'checkpoints'),
        desc=f'width-{width}-layers-{layers}-label-{class_label}'
    )

    earlystoppingcallback = EarlyStopping(
        monitor="train_loss",
        patience=50,
        verbose=True
    )

    model = GeneClassifier(
        input_dim=input_size,
        output_dim=num_labels,
        weights=class_weights,
        params=params,
        weighted_metrics=weighted_metrics,
    )
    
    trainer = pl.Trainer(
        gpus=(1 if torch.cuda.is_available() else 0),
        auto_lr_find=False,
        max_epochs=epochs, 
        gradient_clip_val=0.5,
        logger=wandb_logger,
        callbacks=[
            uploadcallback, 
        ],
    )

    return trainer, model, traindata, valdata 

