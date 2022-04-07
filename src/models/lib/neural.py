from multiprocessing.sharedctypes import Value
from typing import *
import random

import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.functional import accuracy, precision, recall 
from pytorch_tabnet.tab_network import TabNet

import sys, os 
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from helper import seed_everything

# Set all seeds for reproducibility
seed_everything(42)

class GeneClassifier(pl.LightningModule):
    def __init__(self, 
        input_dim: int, 
        output_dim: int, 
        weights: List[torch.Tensor]=None,
        params: Dict[str, float]={
            'width': 1024,
            'layers': 2,
            'lr': 0.001,
            'momentum': 0,
            'weight_decay': 0
        },
        metrics: Dict[str, Callable]={
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
        },
        weighted_metrics=False,
        model=None, 
    ):
        """
        Initialize the gene classifier neural network

        Parameters:
        input_dim: Number of features in the inpute matrix 
        output_dim: Number of classes
        weights: Weights to use in loss calculation to account for imbalance in class size 
        params: Dictionary of hyperparameters to use. Must include width, layers, lr, momentum, weight_decay
        metrics: Dictionary of metrics to log, where keys are metric names and values are torchmetrics.functional methods
        weighted_metrics: If True, use class-weighted calculation in metrics. Otherwise, use default 'micro' calculation.
        """

        super().__init__()
        print(f'Model initialized. {input_dim = }, {output_dim = }. Metrics are {metrics.keys()} and {weighted_metrics = }')

        # save metrics for logging at each step
        self.metrics = metrics

        # Record entire dict for logging
        self._hyperparams = params

        # Record metric calculation scheme
        self.weighted_metrics = weighted_metrics

        # Set hyperparameters
        self.width = params['width']
        self.layers = params['layers']
        self.lr = params['lr']
        self.momentum = params['momentum']
        self.weight_decay = params['weight_decay']

        # Save weights for calculation in loss 
        self.weights = weights

        # Save input/output size, also for metric calculation 
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Generate layers based on width and number of layers 
        if model is not None:
            self.model = model 
        else:
            layers = self.layers*[
                nn.Linear(self.width, self.width),
                nn.ReLU(),
                # nn.Dropout(0.25),
                nn.BatchNorm1d(self.width),
            ]

            # Generate feedforward stack 
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(input_dim, self.width),
                *layers,
                nn.Linear(self.width, output_dim),
            )

    def forward(self, x):
        return (
            self.model(x) if self.model else self.linear_relu_stack(self.flatten(x))
        )

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.lr, 
            momentum=self.momentum, 
            weight_decay=self.weight_decay,
        )

        return optimizer

    # def on_train_start(self):
    #     self.logger.log_hyperparams(self.width)
    #     self.logger.log_hyperparams(self.layers)
    #     self.logger.log_hyperparams(self.lr)
    #     self.logger.log_hyperparams(self.momentum)
    #     self.logger.log_hyperparams(self.weight_decay)

    def _compute_metrics(self, y_hat, y, tag, on_epoch=True, on_step=False):
        for name, metric in self.metrics.items():
            if not self.weighted_metrics: # We dont consider class support in calculation
                val = metric(y_hat, y, average='weighted', num_classes=self.output_dim)
                self.log(
                    f"weighted_{tag}_{name}", 
                    val, 
                    on_epoch=on_epoch, 
                    on_step=on_step,
                    logger=True,
                )
            else:
                val = metric(y_hat, y)
                self.log(
                    f"{tag}_{name}", 
                    val, 
                    on_epoch=on_epoch, 
                    on_step=on_step,
                    logger=True,
                )

    def training_step(self, batch, batch_idx, dataset_idx):
        x, y = batch

        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y, weight=self.weights)

        self.log("train_loss", loss, logger=True, on_epoch=True, on_step=True)
        self._compute_metrics(y_hat, y, 'train')

        return loss
    
    def validation_step(self, batch, batch_idx, dataset_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y, weight=self.weights)

        self.log("val_loss", val_loss, logger=True, on_epoch=True, on_step=True)
        self._compute_metrics(y_hat, y, 'val')
        return val_loss

class TabNetGeneClassifier(TabNet):
    """
    Just a simple wrapper to only return the regular output instead the output and M_loss as defined in the tabnet paper.
    This allows me to use a single train/val/test loop for both models. 
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    # Don't need extra sparsity. See TabNet paper/repository for more information
    def forward(self, x):
        out, _ = super().forward(x)
        return out