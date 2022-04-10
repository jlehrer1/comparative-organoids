from multiprocessing.sharedctypes import Value
from typing import *

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.functional import accuracy, precision, recall 
from pytorch_tabnet.tab_network import TabNet

import sys, os 
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from helper import seed_everything

# Set all seeds for reproducibility
seed_everything(42)

class GeneClassifier(pl.LightningModule):
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
    def __init__(
        self, 
        input_dim, 
        output_dim,
        base_model=None,
        optim_params: Dict[str, float]={
            'optimizer': torch.optim.Adam,
            'lr': 0.001,
            'weight_decay': 0.01,
        },
        metrics: Dict[str, Callable]={
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
        },
        weighted_metrics=False,
        *args,
        **kwargs,
    ):
        super().__init__()
        print(f'Model initialized. {input_dim = }, {output_dim = }. Metrics are {metrics.keys()} and {weighted_metrics = }')

        if base_model is None:
            self.base_model = TabNetGeneClassifier(
                input_dim=input_dim,
                output_dim=output_dim,
                *args,
                **kwargs,
            )
        else:
            self.base_model = base_model
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.optim_params = optim_params
        self.metrics = metrics
        self.weighted_metrics = weighted_metrics
        
    def forward(self, x):
        out = self.base_model(x)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        loss = F.cross_entropy(y_hat, y)
        
        self.log("train_loss", loss, logger=True, on_epoch=True, on_step=True)
        self._compute_metrics(y_hat, y, 'train')
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y)

        self.log("val_loss", val_loss, logger=True, on_epoch=True, on_step=True)
        self._compute_metrics(y_hat, y, 'val')
        
        return val_loss
    
    def _compute_metrics(self, y_hat, y, tag, on_epoch=True, on_step=True):
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

    def configure_optimizers(self):
        optimizer = self.optim_params.pop('optimizer')
        optimizer = optimizer(self.parameters(), **self.optim_params)
        
        return optimizer

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