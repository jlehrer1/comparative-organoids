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
import helper 
from helper import seed_everything, gene_intersection, download

# Set all seeds for reproducibility
seed_everything(42)

class GeneDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        datafiles: List[str],
        labelfiles: List[str],
        class_label: str,
        refgenes: List[str],
        test_prop: float=0.2,
        collocate: bool=False,
        transpose: bool=False,
        batch_size: int=16,
        num_workers=32,
        shuffle=False,
        drop_last: bool=False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.__dict__.update(**kwargs)

        self.datafiles = datafiles
        self.labelfiles = labelfiles
        self.class_label = class_label
        self.refgenes = refgenes
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.test_prop = test_prop
        self.collocate = collocate
        self.transpose = transpose

        self.num_workers = num_workers
        self.batch_size = batch_size
        
        self.trainloaders = []
        self.valloaders = []
        self.testloaders = []
        
        self.args = args
        self.kwargs = kwargs             

    def setup(self, stage: Optional[str] = None):
        print('Creating dataloaders...')
        trainloaders, valloaders, testloaders = generate_loaders(
            datafiles=self.datafiles,
            labelfiles=self.labelfiles,
            class_label=self.class_label,
            refgenes=self.refgenes,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
            collocate=self.collocate, # Join all loaders into one sequential one 
            transpose=self.transpose,
            *self.args,
            **self.kwargs
        )

        print('Done, continuing to training.')

        self.trainloaders = trainloaders
        self.valloaders = valloaders
        self.testloaders = testloaders
        
    def train_dataloader(self):
        return self.trainloaders

    def val_dataloader(self):
        return self.valloaders

    def test_dataloader(self):
        return self.testloaders

# This has to be outside of the datamodule 
# Since we have to download the files to calculate the gene intersection 
def prepare_data(data_path, datafiles, labelfiles):
    os.makedirs(os.path.join(data_path, 'interim'), exist_ok=True)
    os.makedirs(os.path.join(data_path, 'processed', 'labels'), exist_ok=True)

    for datafile, labelfile in zip(datafiles, labelfiles):
        if not os.path.isfile(datafile):
            print(f'Downloading {datafile}')
            download(
                remote_name=os.path.join('jlehrer/expression_data/interim/', datafile.split('/')[-1]),
                file_name=datafile,
            )
        else:
            print(f'{datafile} exists, continuing...')

        if not os.path.isfile(labelfile):
            print(f'Downloading {labelfile}')
            download(
                remote_name=os.path.join('jlehrer/expression_data/labels/', labelfile.split('/')[-1]),
                file_name=labelfile,
            )
        else:
            print(f'{labelfile} exists, continuing...\n')    

def generate_trainer(
    here: str, 
    datafiles: List[str],
    labelfiles: List[str],
    class_label: str,
    num_workers: int=4,
    batch_size: int=4,
    drop_last: bool=True,
    weighted_metrics: bool=False,
    shuffle: bool=False,
    test_prop: float=0.2,
    collocate: bool=False, 
    transpose: bool=False, 
    *args,
    **kwargs,
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

    data_path = os.path.join(here, '..', '..', 'data')

    wandb_logger = WandbLogger(
        project=f"cell-classifier-{class_label}",
        name='TabNet Classifier, Shuffle=True',
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

    prepare_data(
        data_path=data_path,
        datafiles=datafiles,
        labelfiles=labelfiles,
    )

    refgenes = gene_intersection()
    module = GeneDataModule(
        datafiles=datafiles, 
        labelfiles=labelfiles, 
        class_label='Type', 
        refgenes=refgenes,
        skip=3, 
        normalize=True,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=drop_last,
        shuffle=shuffle,
        test_prop=test_prop,
        collocate=collocate,
        transpose=transpose,
        *args,
        **kwargs,
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
            earlystoppingcallback,
        ],
    )

    return trainer, model, module

