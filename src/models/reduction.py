import pandas as pd 
import pathlib 
import os
import umap
import dask.dataframe as da
import dask
import sys
import argparse 
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from helper import upload

# Not sure if this works distributed
from dask.diagnostics import ProgressBar
pbar = ProgressBar()                
pbar.register() # global registration

parser = argparse.ArgumentParser(description='Use UMAP to reduce the dimensionality of the gene expression data')
parser.add_argument(
    '-neighbors', 
    type=int, 
    help='Number of neighbors for UMAP', 
    required=True
)

parser.add_argument(
    '-components', 
    type=int, 
    help='Number of components for UMAP', 
    required=False,
    default=50
)

args = parser.parse_args()

N_COMP = args.components
NEIGHBORS = args.neighbors

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
primary = da.read_csv(os.path.join(here, '..', '..', 'data', 'processed', 'primary.csv'), assume_missing=True)

print('Reading in organoid data')
organoid = da.read_csv(os.path.join(here, '..', '..', 'data', 'processed', 'organoid.csv'), assume_missing=True)

print(f'Calculating UMAP reduction with n_components={N_COMP} and n_neighbors={NEIGHBORS}')
primary_umap = (pd.DataFrame(
    umap_calc(
        data=primary, 
        n_neighbors=NEIGHBORS, 
        n_components=N_COMP).compute()
    )
)

organoid_umap = (pd.DataFrame(
    umap_calc(
        data=organoid,
        n_neighbors=NEIGHBORS,
        n_components=N_COMP).compute()
    )
)

print('Writing to csv')
primary_umap.to_csv(f'primary_reduction_neighbors_{NEIGHBORS}_components_{N_COMP}.csv')
organoid_umap.to_csv(f'organoid_reduction_neighbors_{NEIGHBORS}_components_{N_COMP}.csv')

print('Uploading to S3')

upload(
    f'primary_reduction_neighbors_{NEIGHBORS}_components_{N_COMP}.csv',
    os.path.join('reduced_data', f'primary_reduction_neighbors_{NEIGHBORS}_components_{N_COMP}.csv'), 
)

upload(
   f'organoid_reduction_neighbors_{NEIGHBORS}_components_{N_COMP}.csv',
   os.path.join('reduced_data', f'organoid_reduction_neighbors_{NEIGHBORS}_components_{N_COMP}.csv') 
)