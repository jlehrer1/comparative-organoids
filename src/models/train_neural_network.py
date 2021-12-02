import dask.dataframe as dd
import pandas as pd 
import torch
import linecache 
import csv
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from dask_ml.model_selection import train_test_split, HyperbandSearchCV
import argparse
import pathlib, os 

class GeneExpressionData(Dataset):
    def __init__(self, filename, labelname):
        self._filename = filename
        self._labelname = labelname
        self._total_data = 0
        
        with open(filename, "r") as f:
            self._total_data = len(f.readlines()) - 1
    
    def __getitem__(self, idx):
        if idx == 0:
            return self.__getitem__(1)
        
        line = linecache.getline(self._filename, idx + 1)
        label = linecache.getline(self._labelname, idx + 1)
        
        csv_data = csv.reader([line])
        csv_label = csv.reader([label])
        
        data = [x for x in csv_data][0]
        label = [x for x in csv_label][0]
        
        return (
            torch.from_numpy(np.array([float(x) for x in data])),
            torch.from_numpy(np.array([int(float(x)) for x in label])),
        )
    
    def __len__(self):
        return self._total_data
    
    def num_labels(self):
        return pd.read_csv(self._labelname)['# label'].nunique()
    
    def num_features(self):
        return len(self.__getitem__(0)[0])

class NN(nn.Module):
    def __init__(self, N_features, N_labels):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(in_features=N_features, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=N_labels),
        )
        
    def forward(self, x):
        return self.network(x)
        
if __name__ == "__main__":
    here = pathlib.Path(__file__).parent.absolute()
    data_path = os.path.join(here, '..', '..', 'data')

    argparse = argparse.ArgumentParser(
        usage='Generate a deep neural network classifier on the given data with a particular model structure'
    )
    
    # Read in the data assuming it is transposed (rows are cells, columns are genes)
    primary = dd.read_csv(
        os.path.join(data_path, 'processed', 'primary.csv')
    )

    labels = dd.read_csv(
        os.path.join(data_path, 'processed', 'labels.csv')
    )

    n_labels = dd.loc[:, 'label'].nunique().compute()

    X_train, X_test, y_train, y_test = train_test_split(primary, labels, test_size=0.1)



