import comet_ml
import dask.dataframe as dd
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
from pytorch_lightning.loggers import CometLogger
from sklearn.utils.class_weight import compute_class_weight
from torchmetrics import Accuracy
from helper import upload 

torch.manual_seed(0)
LAYERS = 100
WIDTH = 2048 

class GeneExpressionData(Dataset):
    def __init__(self, filename, labelname):
        self._filename = filename
        self._labelname = labelname
        self._total_data = 0
        
        with open(filename, "r") as f:
            self._total_data = len(f.readlines()) - 1
    
    def __getitem__(self, idx):        
        line = linecache.getline(self._filename, idx + 2)
        label = linecache.getline(self._labelname, idx + 2)
        
        csv_data = csv.reader([line])
        csv_label = csv.reader([label])
        
        data = [x for x in csv_data][0]
        label = [x for x in csv_label][0]
        return torch.from_numpy(np.array([float(x) for x in data])).float(), [int(float(x)) for x in label][0]
    
    def __len__(self):
        return self._total_data
    
    def num_labels(self):
        return pd.read_csv(self._labelname)['# label'].nunique()
    
    def num_features(self):
        return len(self.__getitem__(0)[0])

    def compute_class_weights(self):
        label_df = pd.read_csv(self._labelname)

        weights = compute_class_weight(
            class_weight='balanced', 
            classes=np.unique(label_df), 
            y=label_df.values.reshape(-1))    

        weights = torch.from_numpy(weights)
        return weights

class GeneClassifier(pl.LightningModule):
    def __init__(self, N_features, N_labels, weights, layers):
        """
        Initialize the gene classifier neural network

        Parameters:
        N_features: Number of features in the inpute matrix 
        N_labels: Number of classes 
        """

        super(GeneClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(N_features, 512),
            nn.ReLU(),
            nn.Linear(512, WIDTH),
            nn.ReLU(),
            *layers,
            nn.Linear(WIDTH, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, N_labels),
        )
        
        self.accuracy = Accuracy()
        self.weights = weights

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-3, momentum=0.8)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y, weight=self.weights)
        acc = self.accuracy(y_hat.softmax(dim=-1), y)

        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True)
        self.log("train_accuracy", acc, on_step=True, on_epoch=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y, weight=self.weights)
        acc = self.accuracy(y_hat.softmax(dim=-1), y)

        self.log("val_loss", val_loss, on_step=True, on_epoch=True, logger=True)
        self.log("val_accuracy", acc, on_step=True, on_epoch=True, logger=True)
        return val_loss

class UploadCallback(pl.callbacks.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % 100 == 0: # since we're only saving every 100 epochs
            # Add upload here
            pass

def fix_labels(file, path):
    labels = pd.read_csv(file)
    labels['# label'] = labels['# label'].astype(int) + 1
    labels.to_csv(os.path.join(path, 'fixed_' + file.split('/')[-1]), index=False)


def generate_trainer(here, WIDTH, LAYERS, EPOCHS):

    layers = [
        nn.Linear(WIDTH, WIDTH),
        nn.ReLU(),
        nn.Dropout(0.5), 
    ]

    layers = layers*LAYERS

    data_path = os.path.join(here, '..', '..', 'data', 'processed')
    label_file = 'primary_labels_neighbors_50_components_50_clust_size_100.csv'

    fix_labels(os.path.join(data_path, label_file), here)
    fixed_labels = pd.read_csv(os.path.join(here, f'fixed_{label_file}'))

    dataset = GeneExpressionData(
        filename=os.path.join(data_path, 'primary.csv'),
        labelname=os.path.join(fixed_labels)
    )

    comet_logger = CometLogger(
        api_key="neMNyjJuhw25ao48JEWlJpKRR",
        project_name="gene-expression-classification",  # Optional
        experiment_name=f'Gene Classifier, {LAYERS*3 + 5} Layers w/ Early Stopping'
    )

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train, test = torch.utils.data.random_split(dataset, [train_size, test_size])

    traindata = DataLoader(train, batch_size=8, num_workers=8)
    valdata = DataLoader(test, batch_size=8, num_workers=8)

    earlystopping = pl.callbacks.early_stopping.EarlyStopping(
        monitor='val_loss_epoch',
        patience=50,
    )
    
    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(here, 'checkpoints'),
        filename='classifier-checkpoint-{epoch:02d}',
        every_n_epochs=100,
    )

    model = GeneClassifier(
        N_features=dataset.num_features(),
        N_labels=dataset.num_labels(),
        weights=dataset.compute_class_weights(),
        layers=layers
    )
    
    print(model)
    trainer = pl.Trainer(
        gpus=2, 
        accelerator="ddp",
        auto_lr_find=True, 
        max_epochs=EPOCHS, 
        logger=comet_logger,
        callbacks=[earlystopping],
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
    )

    parser.add_argument(
        '--layers',
        required=False,
        default=5,
        help='Number of deep layers in feedforward neural network'
    )

    parser.add_argument(
        '--epochs',
        required=False,
        default=200000,
        help='Total number of allowable epochs the model is allowed to train for'
    )

    args = parser.parse_args()

    WIDTH, LAYERS, EPOCHS = args.width, args.layers, args.epochs 
    
    trainer, model, traindata, valdata = generate_trainer(here, WIDTH, LAYERS, EPOCHS)
    trainer.fit(model, traindata, valdata)

