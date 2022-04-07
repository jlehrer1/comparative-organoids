import sys
import os
from typing import *

import torch
import wandb 

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from .neural import GeneClassifier
from .train import UploadCallback
from .data import SequentialLoader, generate_loaders

import sys, os 
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from helper import seed_everything, gene_intersection

# Set all seeds for reproducibility
seed_everything(42)

class GeneDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        datafiles: List[str],
        labelfiles: List[str],
        class_label: str,
        refgenes: List[str],
        batch_size: int=16,
        num_workers=32,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.__dict__.update(**kwargs)

        self.datafiles = datafiles
        self.labelfiles = labelfiles
        self.class_label = class_label
        self.refgenes = refgenes
        
        self.num_workers = num_workers
        self.batch_size = batch_size
        
        self.trainloaders = []
        self.valloaders = []
        self.testloaders = []
        
        self.args = args
        self.kwargs = kwargs
        
    def prepare_data(self):
        # Download data from S3 here 
        pass 
    
    def setup(self, stage: Optional[str] = None):
        trainloaders, valloaders, testloaders = generate_loaders(
            datafiles=self.datafiles,
            labelfiles=self.labelfiles,
            class_label=self.class_label,
            refgenes=self.refgenes,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            *self.args,
            **self.kwargs
        )
        
        self.trainloaders = SequentialLoader(trainloaders)
        self.valloaders = SequentialLoader(valloaders)
        self.testloaders = SequentialLoader(testloaders)
        
    def train_dataloader(self):
        return self.trainloaders

    def val_dataloader(self):
        return self.valloaders

    def test_dataloader(self):
        return self.testloaders

def generate_trainer(
    here: str, 
    datafiles: List[str],
    labelfiles: List[str],
    params: Dict[str, float], 
    class_label: str,
    num_workers: int=4,
    batch_size: int=4,
    weighted_metrics: bool=False,
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
    data_path = os.path.join(here, '..', '..', 'data', 'processed')

    wandb_logger = WandbLogger(
        project=f"cell-classifier-{class_label}",
    )

    uploadcallback = UploadCallback(
        path=os.path.join(here, 'checkpoints'),
        desc=f'TabNet Gene Classifier'
    )

    earlystoppingcallback = EarlyStopping(
        monitor="train_loss",
        patience=50,
        verbose=True
    )

    refgenes = gene_intersection()
    module = GeneDataModule(
        datafiles=datafiles, 
        labelfiles=labelfiles, 
        class_label='Type', 
        refgenes=refgenes,
        skip=3, 
        normalize=True,
        batch_size=8,
        num_workers=0,
    )    

    model = GeneClassifier(
        input_dim=len(refgenes),
        output_dim=19,
        weighted_metrics=weighted_metrics,
    )
    
    trainer = pl.Trainer(
        gpus=(1 if torch.cuda.is_available() else 0),
        auto_lr_find=False,
        gradient_clip_val=0.5,
        logger=wandb_logger,
        callbacks=[
            uploadcallback, 
        ],
    )

    return trainer, model, module

