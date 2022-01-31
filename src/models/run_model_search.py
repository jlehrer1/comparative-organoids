import numpy as np 
import random
from itertools import product 
import pathlib 
import os 
import argparse
from scipy.stats import loguniform

def run_search(num, class_label):
    here = pathlib.Path(__file__).parent.absolute()
    yaml_path = os.path.join(here, '..', '..', 'yaml', 'model.yaml')

    class_label = [class_label]
    epochs = [100000]
    lr = np.linspace(0.001, 0.1, 10) #(start, stop, num)
    # momentum = np.linspace(0.001, 0.9, 10)
    momentum = [0]
    # weight_decay = loguniform.rvs(0.001, 0.1, size=10)
    weight_decay = [0]
    width = [64, 128, 1024, 2048]
    layers = np.arange(10, 25, 5)

    params = list(product(width, layers, epochs, lr, momentum, weight_decay, class_label))
    param_names = ['width', 'layers', 'epochs', 'lr', 'momentum', 'weight_decay', 'class_label']
    
    for i, params in enumerate(random.sample(params, num)):
        for n, p in zip(param_names, params):
            os.environ[n.upper()] = str(p)

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
        help='Class to train classifer on',
        required=False,
        default='Subtype',
    )

    args = parser.parse_args()

    run_search(args.N, args.class_label)