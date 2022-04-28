import os
import pathlib 
import sys
import anndata as an
import torch 
import argparse 

from os.path import join, dirname, abspath
sys.path.append(join(dirname(abspath(__file__)), '..', 'src'))

import pandas as pd 
import numpy as np 
import anndata as an 
import sys, os 
sys.path.append('../src')

import sys
import os
import pathlib 
from typing import *

import torch
import numpy as np 
import pandas as pd 
import anndata as an

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from models.lib.data import *
from models.lib.lightning_train import *
from models.lib.neural import *
from data.downloaders.external_download import *
from torchmetrics.functional import *

from helper import download 

parser = argparse.ArgumentParser()
parser.add_argument(
    '--lr',
    type=float,
    default=0.02,
    required=False,
)

parser.add_argument(
    '--weight-decay',
    type=float,
    default=3e-4,
    required=False,
)

parser.add_argument(
    '--name',
    type=str,
    default=None,
    required=False,
)

device = ('cuda:0' if torch.cuda.is_available() else None)

args = parser.parse_args()
lr, weight_decay, name = args.lr, args.weight_decay, args.name

data_path = join(pathlib.Path(__file__).parent.resolve(), '..', 'data', 'dental')

print('Making data folder')
os.makedirs(data_path, exist_ok=True)

for file in ['human_dental_T.h5ad', 'labels_human_dental.tsv']:
    print(f'Downloading {file}')

    if not os.path.isfile(join(data_path, file)):
        download(
            remote_name=join('jlehrer', 'dental', file),
            file_name=join(data_path, file),
        )

class_label = 'cell_type'

module = DataModule(
    datafiles=[join(data_path, 'human_dental_T.h5ad')],
    labelfiles=[join(data_path, 'labels_human_dental.tsv')],
    class_label=class_label,
    sep='\t',
    batch_size=4,
    num_workers=0,
)

module.setup()
model = TabNetLightning(
    input_dim=module.num_features,
    output_dim=module.num_labels,
    weights=total_class_weights([join(data_path, 'labels_human_dental.tsv')], class_label, sep='\t', device=device),
    scheduler_params={
        'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau,
        'factor': 0.001,
    },
    metrics={
        'confusionmatrix': confusion_matrix,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
    },
    weighted_metrics=False,
)

wandb_logger = WandbLogger(
    project=f"tabnet-classifer-sweep",
    name='Dental model local'
)

trainer = pl.Trainer(
    logger=wandb_logger
)

trainer.fit(model, datamodule=module)