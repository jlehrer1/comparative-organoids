import numpy as np
import dask.dataframe as dd
import pathlib 
import os
import torch.nn as nn
import torch.nn.functional as F
from sklearn.svm import SVC
import argparse
from dask_ml.model_selection import train_test_split, RandomizedSearchCV

class GeneClassifier:
    def __init__(self, est, params):
        self.est = est
        self.params = params
        
    def generate_model(self, X, y, n_iter=10):
        grid = RandomizedSearchCV(
            n_iter=n_iter,
            estimator=self.est,
            param_distributions=self.params,
            scoring='balanced_accuracy'
        )

        self.grid = grid.fit(X, y)
    
    def best_score(self):
        return self.grid.best_score_
    
    def best_model(self):
        return self.grid.best_estimator_
    
    def best_params(self):
        return self.grid.best_params_

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

    argparse.add_argument(
        '--minclust',
        required=False,
        type=int,
        help='Min cluster size from HDBSCAN',
        default=250,
    )

    args = argparse.parse_args()
    model = args.model 
    N = args.neighbors 
    COMP = args.components
    MIN_CLUST_SIZE = args.minclust
    
    X = dd.read_csv(
        os.path.join(data_path, 'processed', 'primary.csv')
    )

    y = dd.read_csv(
        os.path.join(
            data_path, 
            'processed', 
            f'primary_labels_neighbors_{N}_components_{COMP}_clust_size_{MIN_CLUST_SIZE}.csv'
        )
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    if model == 'svc':
        from sklearn.svm import SVC
        
        params = {
            'C' : np.linspace(1, 1000, 100),
            'kernel' : ['linear', 'poly', 'rbf', 'sigmoid'],
            'class_weight' : ['balanced']
        }

        svc_est = GeneClassifier(SVC(), params)
        svc_est = svc_est.generate_model(X_train.values, y_train.values)
        print(svc_est)

    elif model == 'tree':
        from xgboost import XGBClassifier

        params = {
            'eta' : np.linspace(0, 1, 20),
            'gamma': np.linspace(0, 1000, 20),
            'max_depth': np.linspace(0, 1000, 20, dtype=int),
        }

        xgb_est = GeneClassifier(XGBClassifier(), params)
        xgb_est = xgb_est.generate_model(X_train.values, y_train.values, n_iter=2)
        print(xgb_est.best_score(), xgb_est.best_params())

    else: # model == 'logistic'
        from dask_ml.linear_model import LogisticRegression
        print('Running logistic model')
        param_distributions = {
            'penalty' : ['l1', 'l2'],
            'C' : np.linspace(0.1, 100, 50)
        }

        logistic_est = GeneClassifier(LogisticRegression(), param_distributions)
        logistic_est = logistic_est.generate_model(X_train.values, y_train.values, n_iter=2)
        print(logistic_est)

