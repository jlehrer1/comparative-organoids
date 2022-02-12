import random
import pathlib 
import os 
import argparse
import ast 
from itertools import product 

import numpy as np 
from scipy.stats import loguniform

def run_search(
    num: int, 
    class_label: str,
    weighted_metrics: bool 
) -> None:
    """
    Runs hyperparameter search by scaling i GPU jobs, i=1,..,num on the PRP Nautilus cluster.

    Parameters:
    num: Number of models to train
    class_label: Which target label to train for 
    weighted_metrics: Whether to use weighted metric calculations or regular ('weighted' vs 'micro' in Torchmetrics)

    """
    here = pathlib.Path(__file__).parent.absolute()
    yaml_path = os.path.join(here, '..', '..', 'yaml', 'model.yaml')

    param_dict = {
        'weighted_metrics': [weighted_metrics],
        'class_label': [class_label],
        'epochs': [100000],
        'lr': np.linspace(0.001, 0.1, 10), #(start, stop, num),
        'batch_size': [2, 4, 16, 32],
        # 'momentum': np.linspace(0.001, 0.9, 10),
        'momentum': [0],
        # 'weight_decay': loguniform.rvs(0.001, 0.1, size=10),
        'weight_decay': [0],
        'width': [1024, 2048, 4096],
        # 'layers': np.arange(10, 25, 5),
        'layers' : [15],
    }

    # Generate cartesian product of dictionary 
    params = list(product(*param_dict.values()))
    param_names = list(param_dict.keys())
    
    for i, params in enumerate(random.sample(params, num)):
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
        default='Subtype',
        type=str,
        help='Class label to train classifier on',
    )

    parser.add_argument(
        '--weighted-metrics',
        type=ast.literal_eval, # To evaluate weighted_metrics=False as an actual bool
        default=False,
        required=False,
        help='Whether to use class-weighted schemes in metric calculations'
    )

    args = parser.parse_args()
    run_search(args.N, args.class_label, args.weighted_accuracy)