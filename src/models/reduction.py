import pandas as pd 
import numpy as np
import pathlib 
import os
import umap
import dask.dataframe as dd
import boto3
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import dask.dataframe as da
import dask
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from helper import upload

# Not sure if this works distributed
from dask.diagnostics import ProgressBar
pbar = ProgressBar()                
pbar.register() # global registration

N_COMP = 100
NEIGHBORS = 500

@dask.delayed
def umap_calc(data, n_neighbors, n_components):
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        verbose=True,
        random_state=42,
    )

    return fit.fit_transform(data)

here = pathlib.Path(__file__).parent.absolute()

print('Reading in primary data')
primary = da.read_csv(os.path.join(here, '..', '..', 'data', 'processed', 'primary.csv'))

print(f'Calculating UMAP reduction with n_components={N_COMP} and n_neighbors={NEIGHBORS}')
primary_umap = pd.DataFrame(umap_calc(primary, NEIGHBORS, N_COMP).compute())

print('Writing to csv')
primary_umap.to_csv(f'primary_reduction_neighbors_{NEIGHBORS}_components_{N_COMP}')

print('Uploading to S3')

upload(
    os.path.join('reduced_data', f'primary_reduction_neighbors_{NEIGHBORS}_components_{N_COMP}.csv'), 
    f'primary_reduction_neighbors_{NEIGHBORS}_components_{N_COMP}.csv'
)