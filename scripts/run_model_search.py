import numpy as np 
import random
from itertools import product 
import pathlib 
import os 
import argparse
from scipy.stats import loguniform


def run_search(num):
    here = pathlib.Path(__file__).parent.absolute()
    yaml_path = os.path.join(here, '..', 'yaml', 'model.yaml')

    epochs = [100000]
    lr = loguniform.rvs(1e-4, 1e-1, size=5)
    momentum = loguniform.rvs(1e-2, 0.8, size=5)
    weight_decay = loguniform.rvs(1e-5, 1e-1, size=10)
    width = [64, 128, 1024, 2048]
    layers = np.arange(10, 25, 5)

    params = list(product(width, layers, epochs, lr, momentum, weight_decay))
    param_samples = random.sample(params, num)
    param_names = [('width', 'layers', 'epochs', 'lr', 'momentum', 'weight_decay')]*num

    for i, (name_sample, param_sample) in enumerate(zip(param_names, param_samples)):
        for n, p in zip(name_sample, param_sample):
            os.environ[n.upper()] = str(p)
        os.environ['I'] = str(i)
        os.system(f'envsubst < {yaml_path} | kubectl create -f -')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(usage='Run random hyperparameter / architecture search')

    parser.add_argument(
        '--N',
        help='Number of experiments to run',
        required=False,
        type=int,
        default=100,
    )

    args = parser.parse_args()
    run_search(args.N)