import pandas as pd 
import sklearn as sk
import numpy as np
import dask.dataframe as dd
import matplotlib.pyplot as plt 
import pathlib 
import os 
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

class GCNetwork(nn.Module):
    def __init__(self, data):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features=100, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=data['label'].nunique()),
        )
        
    def forward(self, x):
        return self.network(x)

class GCSVC:
    def __init__(self):
        self.svc = SVC()
        self.parameters = {'C': [0.1, 0.25, 0.5, 0.75, 1]}

    def generate_model(self, X, y):
        grid = GridSearchCV(
            self.svc,
            self.parameters,
        )

        return grid.fit(X, y).best_estimator_

if __name__ == "__main__":
    here = pathlib.Path(__file__).parent.absolute()
    data_path = os.path.join(here, '..', '..', 'data')

    # Read in the data assuming it is transposed (rows are cells, columns are genes)
    df_primary = dd.read_csv(
        os.path.join(here, '..', '..', 'data', 'processed', 'primary.csv')
    )
