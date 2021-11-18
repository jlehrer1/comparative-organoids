import pandas as pd 
import dask.dataframe as da
from dask.diagnostics import ProgressBar
import numpy as np
import pathlib 
from tqdm import tqdm
import boto3 
import os 
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from helper import download, upload

pbar = ProgressBar()                
pbar.register() # global registration

here = pathlib.Path(__file__).parent.absolute()

if not os.path.isfile(os.path.join(here, '..', '..', 'data', 'interim', 'organoid_T.csv')):
    print('Downloading raw organoid data from S3')
    download(os.path.join('transposed_data', 'organoid_T.csv'), os.path.join(here, '..', '..', 'data', 'interim', 'organoid_T.csv'))

if not os.path.isfile(os.path.join(here, '..', '..', 'data', 'interim', 'primary_T.csv')):
    print('Downloading raw primary data from S3')
    download(os.path.join('transposed_data', 'primary_T.csv'), os.path.join(here, '..', '..', 'data', 'interim', 'primary_T.csv'))

print('Reading in raw organoid data with Dask')
organoid = da.read_csv(os.path.join(here, '..', '..', 'data', 'interim', 'organoid_T.csv'), dtype='float64')

print('Reading in raw primary data with Dask')
primary = da.read_csv(os.path.join(here, '..', '..', 'data', 'interim', 'primary_T.csv'), dtype='float64')

# Fix gene expression names in organoid data
print('Fixing organoid column names')
organoid_cols = [x.split('|')[0] for x in organoid.columns]
organoid.columns = organoid_cols

# Consider only the genes between the two
print('Calculating gene intersection')
subgenes = list(set(organoid.columns).intersection(primary.columns))
print(f'Length of subgenes is {len(subgenes)}')
print(f'Type of organoid and primary is {type(organoid)}, {type(primary)}')

# Just keep those genes
organoid = organoid.loc[:, subgenes]
primary = primary.loc[:, subgenes]

# Fill NaN's with zeros
print('Filling NaN\'s with zeros')
organoid = organoid.fillna(0)
primary = primary.fillna(0)

print('Doing all computations')
organoid = organoid.persist()
primary = primary.persist()

# Write out files 
print('Writing out clean organoid data to csv')
organoid.to_csv(os.path.join(here, '..', '..', 'data', 'processed', 'organoid.csv'), index=False, single_file=True)

print('Writing out clean primary data to csv')
primary.to_csv(os.path.join(here, '..', '..', 'data', 'processed', 'primary.csv'), index=False, single_file=True)

print('Uploading files to S3')
upload(os.path.join(here, '..', '..', 'data', 'processed', 'primary.csv'), 'primary.csv')
upload(os.path.join(here, '..', '..', 'data', 'processed', 'organoid.csv'), 'organoid.csv')