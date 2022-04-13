import random
import pathlib 
import os 
import argparse
import ast 
from itertools import product 

import numpy as np 
from scipy.stats import loguniform

def run_search(
    N: int, 
    class_label: str,
) -> None:
    """
    Runs hyperparameter search by scaling i GPU jobs, i=1,..,N on the PRP Nautilus cluster.

    Parameters:
    N: Number of models to train
    class_label: Which target label to train for 
    weighted_metrics: Whether to use weighted metric calculations or regular ('weighted' vs 'micro' in Torchmetrics)
    """
    
    here = pathlib.Path(__file__).parent.absolute()
    yaml_path = os.path.join(here, '..', '..', 'yaml', 'model.yaml')

    param_dict = {
        'weighted_metrics': [True],
        'class_label': [class_label],
        'max_epochs': [1000],
        'lr': loguniform.rvs(0.001, 0.1, size=10),
        'batch_size': [32],
        'momentum': loguniform.rvs(0.001, 0.9, size=10),
        'weight_decay': loguniform.rvs(0.001, 0.1, size=10),
    }
    
    # Generate cartesian product of dictionary 
    params = list(product(*param_dict.values()))
    param_names = list(param_dict.keys())
    
    for i, params in enumerate(random.sample(params, N)):
        for n, p in zip(param_names, params):
            os.environ[n.upper()] = str(p)

        # These two are to put in job name
        os.environ['NAME'] = class_label.lower()
        os.environ['I'] = str(i) 
        os.system(f'envsubst < {yaml_path} | kubectl create -f -')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(usage='Hyperparameter tune with random search.')

    parser.add_argument(
        '--N',
        help='Number of experiments to run',
        required=False,
        type=int,
        default=100,
    )
    
    parser.add_argument(
        '--class-label',
        required=False,
        default='Type',
        type=str,
        help='Class label to train classifier on',
    )

    args = parser.parse_args()
    args = vars(args)
    run_search(**args)