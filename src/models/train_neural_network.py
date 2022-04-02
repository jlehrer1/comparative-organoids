from email.policy import default
from ssl import Options
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

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from lib.train import generate_trainer

def make_args() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--width',
        required=False,
        default=1024,
        help='Width of deep layers in feedforward neural network',
        type=int,
    )

    parser.add_argument(
        '--layers',
        required=False,
        default=5,
        help='Number of deep layers in feedforward neural network',
        type=int,
    )

    parser.add_argument(
        '--epochs',
        required=False,
        default=200000,
        help='Total number of allowable epochs the model is allowed to train for',
        type=int,
    )

    parser.add_argument(
        '--lr',
        required=False,
        default=1e-4,
        help='Learning rate for model optimizer',
        type=float,
    )

    parser.add_argument(
        '--momentum',
        required=False,
        default=0.1,
        help='Momentum for model optimizer',
        type=float,
    )

    parser.add_argument(
        '--weight-decay',
        required=False,
        default=1e-4,
        help='Weight decay for model optimizer',
        type=float,
    )

    parser.add_argument(
        '--class-label',
        required=False,
        default='Subtype',
        type=str,
        help='Class label to train classifier on',
    )

    parser.add_argument(
        '--batch-size',
        required=False,
        default=4,
        type=int,
        help='Number of samples in minibatch'
    )

    parser.add_argument(
        '--num-workers',
        required=False,
        default=40,
        type=int,
        help='Number of workers in DataLoaders'
    )

    parser.add_argument(
        '--weighted-metrics',
        type=ast.literal_eval, # To evaluate weighted_metrics=False as an actual bool
        default=False,
        required=False,
        help='Whether to use class-weighted schemes in metric calculations'
    )

    return parser

if __name__ == "__main__":
    parser = make_args()
    here = pathlib.Path(__file__).parent.absolute()
    data_path = os.path.join(here, '..', '..', 'data', 'processed')

    args = parser.parse_args()
    params = vars(args)

    trainer, model, traindata, valdata = generate_trainer(
        here=here, 
        params=params,
        class_label=params['class_label'],
        num_workers=params['num_workers'],
        batch_size=params['batch_size'],
        weighted_metrics=params['weighted_metrics'],
        datafiles=[os.path.join(data_path, 'primary.csv')],
        labelfiles=['meta_primary_labels.csv'],
    )
    
    trainer.fit(model, traindata, valdata)