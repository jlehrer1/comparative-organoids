from typing import *
import random

import comet_ml
import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy 

# Set all seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class GeneClassifier(pl.LightningModule):
    def __init__(self, 
        N_features: int, 
        N_labels: int, 
        weights: List[torch.Tensor], 
        params: Dict[str, float],
    ):
        """
        Initialize the gene classifier neural network

        Parameters:
        N_features: Number of features in the inpute matrix 
        N_labels: Number of classes
        weights: Weights to use in accuracy calculation, so we can calculate balanced accuracy to account for uneven class sizes
        params: Dictionary of hyperparameters to use, includes width, layers, lr, momentum, weight_decay
        """
        super(GeneClassifier, self).__init__()

        # Record entire dict for logging
        self._hyperparams = params

        # Set hyperparameters
        self.width = params['width']
        self.layers = params['layers']
        self.lr = params['lr']
        self.momentum = params['momentum']
        self.weight_decay = params['weight_decay']

        layers = self.layers*[
            nn.Linear(self.width, self.width),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.BatchNorm1d(self.width),
        ]

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(N_features, self.width),
            *layers,
            nn.Linear(self.width, N_labels),
        )

        self.accuracy = Accuracy(average='weighted', num_classes=N_labels)
        self.weights = weights

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.lr, 
            momentum=self.momentum, 
            weight_decay=self.weight_decay,
        )

        return optimizer

    def on_train_start(self):
        self.logger.log_hyperparams(self.width)
        self.logger.log_hyperparams(self.layers)
        self.logger.log_hyperparams(self.lr)
        self.logger.log_hyperparams(self.momentum)
        self.logger.log_hyperparams(self.weight_decay)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y, weight=self.weights)
        acc = self.accuracy(y_hat.softmax(dim=-1), y)

        self.log("train_loss", loss, logger=True, on_epoch=True)
        self.log("train_accuracy", acc, logger=True, on_epoch=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y, weight=self.weights)
        acc = self.accuracy(y_hat.softmax(dim=-1), y)

        self.log("val_loss", val_loss, logger=True, on_epoch=True)
        self.log("val_accuracy", acc, logger=True, on_epoch=True)

        return val_loss