import sys
import os
import pathlib 
from typing import *

import torch
import numpy as np 
import pandas as pd 
import anndata as an
import warnings 
from functools import cached_property

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.preprocessing import LabelEncoder 

from .neural import GeneClassifier
from .train import UploadCallback
from .data import generate_dataloaders

import sys, os 
from os.path import join, dirname, abspath 
sys.path.append(join(dirname(abspath(__file__)), '..', '..'))

from helper import gene_intersection, download
from data.downloaders.external_download import download_raw_expression_matrices

here = pathlib.Path(__file__).parent.absolute()

class DataModule(pl.LightningDataModule):
    def __init__(
        self, 
        class_label: str,
        datafiles: List[str]=None,
        labelfiles: List[str]=None,
        urls: Dict[str, List[str]]=None,
        unzip: bool=True,
        sep: str=',',
        datapath: str=None,
        is_assumed_numeric: bool=True,
        *args,
        **kwargs,
    ):
        """
        Creates the DataModule for PyTorch-Lightning training.

        This either takes a dictionary of URLs with the format 
            urls = {dataset_name.extension: 
                        [
                            datafileurl,
                            labelfileurl,
                        ]
                    }

        OR two lists containing the absolute paths to the datafiles and labelfiles, respectively.

        :param class_label: Class label to train on. Must be in all label files 
        :type class_label: str
        :param datafiles: List of absolute paths to datafiles, if not using URLS. defaults to None
        :type datafiles: List[str], optional
        :param labelfiles: List of absolute paths to labelfiles, if not using URLS. defaults to None
        :type labelfiles: List[str], optional
        :param urls: Dictionary of URLS to download, as specified in the above docstring, defaults to None
        :type urls: Dict[str, List[str, str]], optional
        :param unzip: Boolean, whether to unzip the datafiles in the url, defaults to False
        :type unzip: bool, optional
        :param sep: Separator to use in reading in both datafiles and labelfiles. WARNING: Must be homogeneous between all datafile and labelfiles, defaults to '\t'
        :type sep: str, optional
        :param datapath: Path to local directory to download datafiles and labelfiles to, if using URL. defaults to None
        :type datapath: str, optional
        :param is_assumed_numeric: If the class_label column in all labelfiles is numeric. Otherwise, we automatically apply sklearn.preprocessing.LabelEncoder to the intersection of all possible labels, defaults to True
        :type is_assumed_numeric: bool, optional
        :raises ValueError: If both a dictionary of URL's is passed and labelfiles/datafiles are passed. We can only handle one, not a mix of both, since there isn't a way to determine easily if a string is an external url or not. 

        """    
        super().__init__()

        # Make sure we don't have datafiles/labelfiles AND urls at start
        if urls is not None and datafiles is not None or urls is not None and labelfiles is not None:
            raise ValueError("Either a dictionary of data to download, or paths to datafiles and labelfiles are supported, but not both.")


        self.class_label = class_label
        self.urls = urls 
        self.unzip = unzip 
        self.sep = sep 
        self.datapath = (
            datapath if datapath is not None else join(here, '..', '..', '..', 'data', 'raw')
        )
        self.is_assumed_numeric = is_assumed_numeric

        # If we have a list of urls, we can generate the list of paths of datafiles/labelfiles that will be downloaded after self.prepare_data()
        if self.urls is not None:
            self.datafiles = [join(self.datapath, f) for f in self.urls.keys()] 
            self.labelfiles = [join(self.datapath, f'labels_{f}') for f in self.urls.keys()] 
        else:
            self.datafiles = datafiles 
            self.labelfiles = labelfiles

        # Warn user in case tsv/csv ,/\t don't match, this can be annoying to diagnose
        if (sep == '\t' and pathlib.Path(labelfiles[0]).suffix == '.csv') or (sep == ',' and pathlib.Path(labelfiles[0]).suffix == '.tsv'):
            warnings.warn(f'Passed delimiter {sep = } doesn\'t match file extension, continuing...')

        self.args = args 
        self.kwargs = kwargs
        
    def prepare_data(self):
        if self.urls is not None:
            download_raw_expression_matrices(
                self.urls,
                unzip=self.unzip,
                sep=self.sep,
                datapath=self.datapath,
            )
        
        if not self.is_assumed_numeric:
            print('is_assumed_numeric=False, using sklearn.preprocessing.LabelEncoder and encoding target variables.')

            unique_targets = list(
                set(np.concatenate([pd.read_csv(df, sep=self.sep).loc[:, self.class_label].unique() for df in self.labelfiles]))
            )
            
            le = LabelEncoder()
            le = le.fit(unique_targets)
            
            for file in self.labelfiles:
                labels = pd.read_csv(file, sep=self.sep)

                labels.loc[:, f'categorical_{self.class_label}'] = labels.loc[:, self.class_label]

                labels.loc[:, self.class_label] = le.transform(
                    labels.loc[:, f'categorical_{self.class_label}']
                )

                labels.to_csv(file, index=False, sep=self.sep) # Don't need to re-index here 

    def setup(self, stage: Optional[str] = None):
        print('Creating train/val/test DataLoaders...')
        trainloader, valloader, testloader = generate_dataloaders(
            datafiles=self.datafiles,
            labelfiles=self.labelfiles,
            class_label=self.class_label,
            sep=self.sep,
            *self.args,
            **self.kwargs,
            pin_memory=True, # For gpu training
        )

        print('Done, continuing to training.')
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        
    def train_dataloader(self):
        return self.trainloader

    def val_dataloader(self):
        return self.valloader

    def test_dataloader(self):
        return self.testloader

    @cached_property
    def num_labels(self):
        val = []
        for file in self.labelfiles:
            val.append(pd.read_csv(file, sep=self.sep).loc[:, self.class_label].values.max())

        return max(val) + 1

    @cached_property
    def num_features(self):
        if 'refgenes' in self.kwargs:
            return len(self.kwargs['refgenes'])
        elif hasattr(self, 'trainloader'):
            return next(iter(self.trainloader))[0].shape[1]
        elif pathlib.Path(self.datafiles[0]).suffix == '.h5ad':
            return an.read_h5ad(self.datafiles[0]).X.shape[1]
        else:
            return pd.read_csv(self.datafiles[0], nrows=1, sep=self.sep).shape[1]
    
# This has to be outside of the datamodule 
# Since we have to download the files to calculate the gene intersection 
def prepare_data(
    data_path: str, 
    datafiles: List[str], 
    labelfiles: List[str],
) -> None:
    """
    Prepare data for model training, by downloading the transposed and clean labels from the S3 bucket

    :param data_path: Path to the top-level folder containing the data subfolders
    :type data_path: str
    :param datafiles: List of absolute paths to datafiles 
    :type datafiles: List[str]
    :param labelfiles: List of absolute paths to labelfiles
    :type labelfiles: List[str]
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
    batch_size: int,
    num_workers: int,
    optim_params: Dict[str, Any]={
        'optimizer': torch.optim.Adam,
        'lr': 0.02,
    },
    weighted_metrics: bool=None,
    scheduler_params: Dict[str, float]=None,
    wandb_name: str=None,
    weights: torch.Tensor=None,
    max_epochs=500,
    *args,
    **kwargs,
):
    """
    Generates PyTorch Lightning trainer and datasets for model training.

    :param datafiles: List of absolute paths to datafiles
    :type datafiles: List[str]
    :param labelfiles: List of absolute paths to labelfiles
    :type labelfiles: List[str]
    :param class_label: Class label to train on 
    :type class_label: str
    :param weighted_metrics: To use weighted metrics in model training 
    :type weighted_metrics: bool
    :param batch_size: Batch size in dataloader
    :type batch_size: int
    :param num_workers: Number of workers in dataloader
    :type num_workers: int
    :param optim_params: Dictionary defining optimizer and any needed/optional arguments for optimizer initializatiom
    :type optim_params: Dict[str, Any]
    :param wandb_name: Name of run in Wandb.ai, defaults to ''
    :type wandb_name: str, optional
    :return: Trainer, model, datamodule 
    :rtype: Trainer, model, datamodule 
    """

    device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device is {device}')
    
    here = pathlib.Path(__file__).parent.absolute()
    data_path = os.path.join(here, '..', '..', '..', 'data')

    wandb_logger = WandbLogger(
        project=f"tabnet-classifer-sweep",
        name=wandb_name
    )

    uploadcallback = UploadCallback(
        path=os.path.join(here, 'checkpoints'),
        desc=wandb_name
    )

    # earlystoppingcallback = EarlyStopping(
    #     monitor="val_loss",
    #     patience=50,
    #     verbose=True
    # )

    prepare_data(
        data_path=data_path,
        datafiles=datafiles,
        labelfiles=labelfiles,
    )

    module = DataModule(
        datafiles=datafiles, 
        labelfiles=labelfiles, 
        class_label=class_label, 
        batch_size=batch_size,
        num_workers=num_workers,
        *args,
        **kwargs,
    )

    model = GeneClassifier(
        input_dim=module.num_features,
        output_dim=module.num_labels,
        weighted_metrics=weighted_metrics,
        optim_params=optim_params,
        scheduler_params=scheduler_params,
        weights=weights,
    )
    
    trainer = pl.Trainer(
        gpus=(1 if torch.cuda.is_available() else 0),
        auto_lr_find=False,
        # gradient_clip_val=0.5,
        logger=wandb_logger,
        max_epochs=max_epochs,
        # callbacks=[
        #     uploadcallback, 
        # ],
        # val_check_interval=0.25, # Calculate validation every quarter epoch instead of full since dataset is large, and would like to test this 
    )

    return trainer, model, module

