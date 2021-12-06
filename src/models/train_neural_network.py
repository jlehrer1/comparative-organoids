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

torch.manual_seed(0)
LAYERS = 5

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

def fix_labels(file, path):
    labels = pd.read_csv(file)
    labels['# label'] = labels['# label'].astype(int) + 1
    labels.to_csv(os.path.join(path, 'fixed_' + file.split('/')[-1]), index=False)

layers = [
    nn.Linear(1024, 1024),
    nn.ReLU(),
    nn.Dropout(np.random.choice([0.1, 0.25, 0.5])), 
]

layers = layers*LAYERS

class GeneClassifier(pl.LightningModule):
    def __init__(self, N_features, N_labels, weights=None):
        """
        Initialize the gene classifier neural network

        Parameters:
        N_features: Number of features in the inpute matrix 
        N_labels: Number of classes 
        weights: Class weights in the case of an unbalanced dataset
        """

        super(GeneClassifier, self).__init__()
        self.weights = weights
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(N_features, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            *layers,
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, N_labels),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def accuracy(self, y_hat, y):
        y_hat = torch.argmax(y_hat, dim=1)
        accuracy = torch.sum(y == y_hat).item() / (len(y) * 1.0)
        return torch.tensor(accuracy)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-3, momentum=0.8)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y, weight=self.weights)
        acc = self.accuracy(y_hat, y)

        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True)
        self.log("train_accuracy", acc, on_step=True, on_epoch=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y, weight=self.weights)
        acc = self.accuracy(y_hat, y)

        self.log("val_loss", val_loss, on_step=True, on_epoch=True, logger=True)
        self.log("val_accuracy", acc, on_step=True, on_epoch=True, logger=True)
        return val_loss

def class_weights(label_df):
    label_df = pd.read_csv(label_df)

    weights = compute_class_weight(
        class_weight='balanced', 
        classes=np.unique(label_df), 
        y=label_df.values.reshape(-1))    

    weights = torch.from_numpy(weights)
    return weights.float().to('cuda')


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device = {device}')

    here = pathlib.Path(__file__).parent.absolute()
    data_path = os.path.join(here, '..', '..', 'data', 'processed')

    fix_labels(os.path.join(data_path, 'primary_labels_neighbors_50_components_50_clust_size_100.csv'), here)
    fixed_labels = pd.read_csv(os.path.join(here, 'fixed_primary_labels_neighbors_50_components_50_clust_size_100.csv'))

    t = GeneExpressionData(
        filename=os.path.join(data_path, 'primary.csv'),
        labelname=os.path.join(here, 'fixed_primary_labels_neighbors_50_components_50_clust_size_100.csv')
    )

    comet_logger = CometLogger(
        api_key="neMNyjJuhw25ao48JEWlJpKRR",
        project_name="gene-expression-classification",  # Optional
        experiment_name=f'Gene Classifier, {LAYERS*3 + 5} Layers w/ Class Weights & Dropout'
    )

    train_size = int(0.8 * len(t))
    test_size = len(t) - train_size

    train, test = torch.utils.data.random_split(t, [train_size, test_size])

    traindata = DataLoader(train, batch_size=8, num_workers=8)
    valdata = DataLoader(test, batch_size=8, num_workers=8)

    weights = class_weights(os.path.join(here, 'fixed_primary_labels_neighbors_50_components_50_clust_size_100.csv'))

    model = GeneClassifier(
        N_features=t.num_features(),
        N_labels=t.num_labels(),
        weights=weights
    )

    epochs = 2000000 # 2 million
    trainer = pl.Trainer(gpus=1, auto_lr_find=True, max_epochs=epochs, logger=comet_logger)
    trainer.fit(model, traindata, valdata)