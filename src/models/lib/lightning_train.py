import sys
import os
import pathlib 
from typing import *

import torch
import wandb 

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from .neural import GeneClassifier
from .train import UploadCallback
from .data import generate_dataloaders

import sys, os 
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import helper 
from helper import seed_everything, gene_intersection, download

# Set all seeds for reproducibility
seed_everything(42)

class GeneDataModule(pl.LightningDataModule):
    """
    Creates the DataModule for PyTorch-Lightning training.

    Parameters:
    **kwargs: The dictionary of keyword-based arguments is passed directly to the generate_dataloaders method, so treat the initialization of
    this class in the same way the user generates the train, val, test dataloaders.
    """
    def __init__(
        self, 
        *args,
        **kwargs,
    ):
        super().__init__()
        self.args = args 
        self.kwargs = kwargs
        
    def setup(self, stage: Optional[str] = None):
        print('Creating train/val/test DataLoaders...')
        trainloaders, valloaders, testloaders = generate_dataloaders(
            *self.args,
            **self.kwargs,
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
def prepare_data(
    data_path: str, 
    datafiles: List[str], 
    labelfiles: List[str],
) -> None:
    """
    Prepare data for model training, by downloading the transposed and clean labels from the S3 bucket

    Parameters:
    data_path: Path to the top-level folder containing the data subfolders
    datafiles: List of absolute paths to datafiles 
    labelfiles: List of absolute paths to labelfiles
    """
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
    datafiles: List[str],
    labelfiles: List[str],
    class_label: str,
    weighted_metrics: bool,
    batch_size: int,
    num_workers: int,
    optim_params: Dict[str, Any],
    wandb_name='',
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
    
    here = pathlib.Path(__file__).parent.absolute()
    data_path = os.path.join(here, '..', '..', '..', 'data')

    wandb_logger = WandbLogger(
        project=f"cell-classifier-{class_label.lower()}",
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
        class_label=class_label, 
        refgenes=refgenes,
        batch_size=batch_size,
        num_workers=num_workers,
        *args,
        **kwargs,
    )

    model = GeneClassifier(
        input_dim=len(refgenes),
        output_dim=19,
        weighted_metrics=weighted_metrics,
        optim_params=optim_params
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
        max_epochs=kwargs['max_epochs']
    )

    return trainer, model, module

