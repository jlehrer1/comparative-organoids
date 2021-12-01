import numpy as np
import dask.dataframe as dd
import pathlib 
import os 
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.svm import SVC
import dask
import argparse

from dask_ml.model_selection import train_test_split, RandomizedSearchCV

class GCDaskEst:
    def __init__(self, est, params) -> None:
        self.est = est
        self.params = params
    
    def generate_model(self, X, y, n_iter=10):
        grid = RandomizedSearchCV(
            n_iter=n_iter,
            estimator=self.svc,
            param_distributions=self.parameters,
            scoring='balanced_accuracy'
        )

        result = grid.fit(X, y)
        return result.best_score_, result.best_params_

if __name__ == "__main__":
    here = pathlib.Path(__file__).parent.absolute()
    data_path = os.path.join(here, '..', '..', 'data')

    argparse = argparse.ArgumentParser(usage='Generate a classifier on the given data with a particular model structure')
    argparse.add_argument(
        '--model',
        required=True,
        choices=['svc', 'tree', 'logistic'],
        help='Architecture of model'
    )

    argparse.add_argument(
        '--neighbors',
        required=False,
        type=int,
        help='Number of neighbors in UMAP',
        default=500,
    )

    argparse.add_argument(
        '--components',
        required=False,
        type=int,
        help='Number of components in UMAP',
        default=100,
    )

    args = argparse.parse_args()
    model = args.model 
    N = args.neighbors 
    COMP = args

    X = dd.read_csv(os.path.join(data_path, 'processed', 'primary.csv'))
    y = dd.read_csv(os.path.join(data_path, 'processed', f'primary_labels_neighbors_{N}_components_{COMP}.csv'))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    if model == 'svc':
        pass
    elif model == 'tree':
        pass
    else: # model == logistic
        pass


