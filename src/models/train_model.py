import numpy as np
import dask.dataframe as dd
import pathlib 
import os 
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import dask
from dask_ml.model_selection import train_test_split

class GCNetwork(nn.Module):
    def __init__(self, N_labels):
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
            nn.Linear(in_features=16, out_features=N_labels),
        )
        
    def forward(self, x):
        return self.network(x)

class GCSVC:
    def __init__(self):
        self.svc = SVC()
        self.parameters = {'C': [0.1, 0.25, 0.5, 0.75, 1]}

    @dask.delayed
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
    primary = dd.read_csv(
        os.path.join(data_path, 'processed', 'primary.csv')
    )

    labels = dd.read_csv(
        os.path.join(data_path, 'processed', 'labels.csv')
    )
    n_labels = dd.loc[:, 'label'].nunique().compute()

    X_train, X_test, y_train, y_test = train_test_split(primary, labels, test_size=0.1)

    nn = GCNetwork(X_train, n_labels)
    svc = GCSVC()
    

