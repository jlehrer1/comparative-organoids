import pathlib 
import os
from pydoc import describe 
import sys
import pandas as pd 
import numpy as np 
import argparse

from sklearn.preprocessing import LabelEncoder
import dask.dataframe as da
from dask.diagnostics import ProgressBar

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from helper import upload, S3_CLEAN_DATA_PATH
from download_data import download_interim
from data_methods import clean_datasets, combine_labelsets

pbar = ProgressBar()
pbar.register() # global registration

here = pathlib.Path(__file__).parent.absolute()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--labels',
        required=False,
        default=None,
        help='If passed, run the code for label cleaning, otherwise, continue.',
        action='store_true',
    )

    parser.add_argument(
        '--data',
        required=False,
        default=None,
        help='If passed, run the code for data cleaning. This should be run remotely, as it requires the interaction between large Dask dataframes which will likely be quite slow. Otherwise, continue',
        action='store_true',
    )

    args = parser.parse_args()
    labels, data = args.labels, args.data 

    if not labels and not data:
        print('Nothing arguments passed. Done.')

    if data: clean_datasets()
    if labels: combine_labelsets()


