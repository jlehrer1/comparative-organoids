import comet_ml
from pytorch_lightning.loggers import CometLogger
import pandas as pd 
import torch
import linecache 
import csv
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse
import pathlib, os
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.utils.class_weight import compute_class_weight
from torchmetrics import Accuracy, ConfusionMatrix 

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from helper import upload 

# torch.manual_seed(0)
class GeneExpressionData(Dataset):
    """
    Defines a PyTorch Dataset for a CSV too large to fit in memory. 

    Parameters:
    filename: Path to csv data file, where rows are samples and columns are features
    labelname: Path to label file, where column '# labels' defines classification labels
    """
    def __init__(self, filename, labelname, class_label):
        self._filename = filename
        self._labelname = pd.read_csv(labelname)
        self._total_data = 0
        self._class_label = class_label
        
        with open(filename, "r") as f:
            self._total_data = len(f.readlines()) - 1
    
    def __getitem__(self, idx):
        line = linecache.getline(self._filename, idx + 2)
        csv_data = csv.reader([line])
        data = [x for x in csv_data][0]
        
        label = self._labelname.loc[idx, self._class_label]
        return torch.from_numpy(np.array([float(x) for x in data])).float(), label
    
    def __len__(self):
        return self._total_data
    
    def num_labels(self):
        return self._labelname[self._class_label].nunique()
    
    def num_features(self):
        return len(self.__getitem__(0)[0])

    def compute_class_weights(self):
        weights = compute_class_weight(
            class_weight='balanced', 
            classes=np.unique(self._labelname[self._class_label].values), 
            y=self._labelname[self._class_label].values
        )

        weights = torch.from_numpy(weights)
        return weights.float().to('cuda')

class GeneClassifier(pl.LightningModule):
    def __init__(self, N_features, N_labels, weights, params):
        """
        Initialize the gene classifier neural network

        Parameters:
        N_features: Number of features in the inpute matrix 
        N_labels: Number of classes
        weights: Weights to use in accuracy calculation, so we can calculate balanced accuracy to account for uneven class sizes
        params: Dictionary of hyperparameters to use, includes width, layers, lr, momentum, weight_decay
        """
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
            nn.Dropout(0.5),
            nn.BatchNorm1d(self.width),
        ]

        super(GeneClassifier, self).__init__()
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
        # matrix = self.confusion(y_hat.softmax(dim=-1), y).cpu().detach().numpy()

        self.log("val_loss", val_loss, logger=True, on_epoch=True)
        self.log("val_accuracy", acc, logger=True, on_epoch=True)
        # self.logger.experiment.log_confusion_matrix(
        #     title="val_confusion_matrix",
        #     matrix=matrix
        # )

        return val_loss

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

def fix_labels(file, path, class_label='# label'):
    """
    Fixes label output from HDBSCAN to be non-negative, since noise points are classified with label -1. PyTorch requires indexing from 0. 

    Parameters:
    file: Path to label file
    path: Path to write corrected label file to
    """
    labels = pd.read_csv(file)
    labels[class_label] = labels[class_label].astype(int) + 1
    labels.to_csv(os.path.join(path, 'fixed_' + file.split('/')[-1]))

def generate_trainer(here, params):
    """
    Generates PyTorch Lightning trainer and datasets for model training.

    Parameters:
    here: Absolute path to __file__
    params: Dictionary of hyperparameters for model training

    Returns:
    Tuple[trainer, model, traindata, valdata]: Tuple of PyTorch-Lightning trainer, model instance, and train and validation dataloaders for training.
    """

    width = params['width']
    epochs = params['epochs']
    layers = params['layers']

    data_path = os.path.join(here, '..', '..', 'data', 'processed')
    label_file = 'meta_primary_labels.csv'

    # fix_labels(os.path.join(data_path, label_file), here) # dont need to fix when using meta primary labels

    dataset = GeneExpressionData(
        filename=os.path.join(data_path, 'primary.csv'),
        labelname=os.path.join(os.path.join(data_path, label_file)),
        class_label='# label'
    )

    comet_logger = CometLogger(
        api_key="neMNyjJuhw25ao48JEWlJpKRR",
        project_name="gene-expression-",  # Optional
        workspace="jlehrer1",
        experiment_name=f'{layers + 5} Layers, {width} Width'
    )

    train_size = int(0.80 * len(dataset))
    test_size = len(dataset) - train_size

    train, test = torch.utils.data.random_split(dataset, [train_size, test_size])

    traindata = DataLoader(train, batch_size=8, num_workers=100)
    valdata = DataLoader(test, batch_size=8, num_workers=100)

    earlystoppingcallback = pl.callbacks.early_stopping.EarlyStopping(
        monitor='val_loss',
        patience=50,
    )
    
    uploadcallback = UploadCallback(
        path=os.path.join(here, 'checkpoints'),
        desc=f'width-{width}-layers-{layers}'
    )

    model = GeneClassifier(
        N_features=dataset.num_features(),
        N_labels=dataset.num_labels(),
        weights=dataset.compute_class_weights(),
        params=params,
    )
    
    print(model)
    trainer = pl.Trainer(
        gpus=1, 
        auto_lr_find=True,
        max_epochs=epochs, 
        logger=comet_logger,
        callbacks=[
            uploadcallback,
        ],
    )

    return trainer, model, traindata, valdata 

def add_args():
    """
    Sets up the argparser for model training
    """
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

    return parser

if __name__ == "__main__":
    here = pathlib.Path(__file__).parent.absolute()

    parser = add_args()
    args = parser.parse_args()

    params = vars(args)
    
    trainer, model, traindata, valdata = generate_trainer(here, params)
    trainer.fit(model, traindata, valdata)

