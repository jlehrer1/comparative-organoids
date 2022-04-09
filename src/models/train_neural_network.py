import sys
import argparse
import pathlib
import os
import ast
from typing import *
from lib.lightning_train import generate_trainer

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import helper
from helper import seed_everything

seed_everything(42)

def make_args() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--width',
        required=False,
        default=1024,
        help='Width of deep layers in feedforward neural network',
        type=int,
    )

    parser.add_argument(
        '--layers',
        required=False,
        default=5,
        help='Number of deep layers in feedforward neural network',
        type=int,
    )

    parser.add_argument(
        '--epochs',
        required=False,
        default=200000,
        help='Total number of allowable epochs the model is allowed to train for',
        type=int,
    )

    parser.add_argument(
        '--lr',
        required=False,
        default=1e-4,
        help='Learning rate for model optimizer',
        type=float,
    )

    parser.add_argument(
        '--momentum',
        required=False,
        default=0.1,
        help='Momentum for model optimizer',
        type=float,
    )

    parser.add_argument(
        '--weight-decay',
        required=False,
        default=1e-4,
        help='Weight decay for model optimizer',
        type=float,
    )

    parser.add_argument(
        '--class-label',
        required=False,
        default='Subtype',
        type=str,
        help='Class label to train classifier on',
    )

    parser.add_argument(
        '--batch-size',
        required=False,
        default=4,
        type=int,
        help='Number of samples in minibatch'
    )

    parser.add_argument(
        '--num-workers',
        required=False,
        default=40,
        type=int,
        help='Number of workers in DataLoaders'
    )

    parser.add_argument(
        '--weighted-metrics',
        type=ast.literal_eval, # To evaluate weighted_metrics=False as an actual bool
        default=False,
        required=False,
        help='Whether to use class-weighted schemes in metric calculations'
    )

    return parser

if __name__ == "__main__":
    parser = make_args()
    here = pathlib.Path(__file__).parent.absolute()

    data_path = os.path.join(here, '..', '..', 'data', 'interim')
    label_path = os.path.join(here, '..', '..', 'data', 'processed', 'labels')

    args = parser.parse_args()
    params = vars(args)

    info = helper.INTERIM_DATA_AND_LABEL_FILES_LIST

    datafiles = info.keys()
    labelfiles = [info[file] for file in datafiles]

    datafiles = [os.path.join(data_path, f) for f in datafiles]
    labelfiles = [os.path.join(label_path, f) for f in labelfiles]
    class_label = 'Type'

    trainer, model, module = generate_trainer(
        datafiles=datafiles,
        labelfiles=labelfiles,
        here=here, 
        class_label=class_label,
        weighted_metrics=True,
        num_workers=16,
        batch_size=4,
        shuffle=True,
    )
    
    trainer.fit(model, datamodule=module)