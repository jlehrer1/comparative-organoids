from ssl import Options
import random
import sys
import argparse
import pathlib
import os
from typing import *

import comet_ml
import pandas as pd 
import torch
import numpy as np
import pytorch_lightning as pl
import wandb 

from pytorch_lightning.loggers import CometLogger, WandbLogger
from pytorch.callbacks import EarlyStopping
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from helper import upload 
from lib.neural import GeneClassifier
from lib.data import GeneExpressionData, generate_datasets
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# Set all seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class UploadCallback(pl.callbacks.Callback):
    """Custom PyTorch callback for uploading model checkpoints to the braingeneers S3 bucket.
    
    Parameters:
    path: Local path to folder where model checkpoints are saved
    desc: Description of checkpoint that is appended to checkpoint file name on save
    upload_path: Subpath in braingeneersdev/jlehrer/ to upload model checkpoints to
    """
    
    def __init__(self, path, desc, upload_path='model_checkpoints') -> None:
        super().__init__()
        self.path = path 
        self.desc = desc
        self.upload_path = upload_path

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % 10 == 0: # Save every ten epochs
            checkpoint = f'checkpoint-{epoch}-desc-{self.desc}.ckpt'
            trainer.save_checkpoint(os.path.join(self.path, checkpoint))
            print(f'Uploading checkpoint at epoch {epoch}')
            upload(
                os.path.join(self.path, checkpoint),
                os.path.join('jlehrer', self.upload_path, checkpoint)
            )

# reproducibility over all workers
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def train_model(
    params: Dict[str, Union[int, float]],
    dataset_files,
    label_files,
    class_label,
    num_workers: int=100,
    batch_size: int=8,
):
    width = params['width']
    epochs = params['epochs']
    layers = params['layers']

    train, test, input_size, num_labels, weights = generate_datasets(dataset_files, label_files, class_label)
    
    g = torch.Generator()
    g.manual_seed(42)

    trainloader = DataLoader(
        train, 
        batch_size=batch_size, 
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
    )

    valloader = DataLoader(
        test, 
        batch_size=batch_size, 
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
    )

    # criterion = 
    model = GeneClassifier(input_size, num_labels, weights)

    # for e in range(epochs):
    #     train_loss = 0.0
    #     model.train()     # Optional when not using Model Specific layer
    #     for data, labels in trainloader:
    #         optimizer.zero_grad()
    #         target = model(data)
    #         loss = criterion(target,labels)
    #         loss.backward()
    #         optimizer.step()
    #         train_loss += loss.item()
        
    #     valid_loss = 0.0
    #     model.eval() # Optional when not using Model Specific layer
    #     for data, labels in valloader:
    #         target = model(data)
    #         loss = criterion(target,labels)
    #         valid_loss = loss.item() * data.size(0)

    #     print(f'Epoch {e+1} \t\t Training Loss: {train_loss / len(trainloader)} \t\t Validation Loss: {valid_loss / len(validloader)}')
    #     if min_valid_loss > valid_loss:
    #         print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
    #         min_valid_loss = valid_loss
    #         # Saving State Dict
    #         torch.save(model.state_dict(), 'saved_model.pth')

def generate_trainer(
    here: str, 
    params: Dict[str, float], 
    label_file: str='meta_primary_labels.csv',
    class_label: str='Subtype',
    num_workers: int=100,
    batch_size: int=8,
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

    # comet_logger = CometLogger(
    #     api_key="neMNyjJuhw25ao48JEWlJpKRR",
    #     project_name=f"cell-classifier-{class_label}",  # Optional
    #     workspace="jlehrer1",
    #     experiment_name=f'{layers + 5} Layers, {width} Width'
    # )

    wandb_logger = WandbLogger(
        project=f"cell-classifier-{class_label}",
        name=f'{layers + 5} Layers, {width} Width'
    )

    train, test, input_size, num_labels, class_weights = generate_datasets(
        dataset_files=[os.path.join(data_path, 'primary.csv')], # TODO: add this list of files as a parameter that can be passed to the training script, test this for now 
        label_files=[os.path.join(data_path, label_file)],
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
        N_features=input_size,
        N_labels=num_labels,
        weights=class_weights,
        params=params,
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

if __name__ == "__main__":
    here = pathlib.Path(__file__).parent.absolute()

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

    args = parser.parse_args()
    params = vars(args)

    trainer, model, traindata, valdata = generate_trainer(
        here=here, 
        params=params,
        label_file='meta_primary_labels.csv',
        class_label=params['class_label'],
        num_workers=params['num_workers'],
        batch_size=params['batch_size'],
    )
    
    trainer.fit(model, traindata, valdata)