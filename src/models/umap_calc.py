import pandas as pd 
import pathlib 
import os
import umap
import dask.dataframe as da
import dask
import sys
import argparse 
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from helper import upload, S3_UMAP_PATH

# Not sure if this works distributed
from dask.diagnostics import ProgressBar
pbar = ProgressBar()                
pbar.register() # global registration


@dask.delayed
def umap_calc(data, n_neighbors, n_components):
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        verbose=True,
        random_state=42,
    )

    return fit.fit_transform(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Use UMAP to reduce the dimensionality of the gene expression data')
    parser.add_argument(
        '-neighbors', 
        type=int, 
        help='Number of neighbors for UMAP', 
        required=True
    )

    parser.add_argument(
        '-file',
        type=str,
        help='One of organoid or primary, which dataset to run umap dimensionality reduction on.',
        choices=['organoid', 'primary'],
        required=True, # We do need to specify!
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
    FILE = args.file

    here = pathlib.Path(__file__).parent.absolute()
    fname = f'{FILE}_reduction_neighbors_{NEIGHBORS}_components_{N_COMP}.csv'

    print(f'Reading in {FILE} data')
    data = da.read_csv(os.path.join(here, '..', '..', 'data', 'processed', f'{FILE}.csv'), assume_missing=True)

    print(f'Calculating UMAP reduction with n_components={N_COMP} and n_neighbors={NEIGHBORS}')
    umap_reduction = (pd.DataFrame(
        umap_calc(
            data=data, 
            n_neighbors=NEIGHBORS, 
            n_components=N_COMP,
            min_dist=0,
            ).compute()
        )
    )

    print(f'Writing {FILE} umap data to csv')
    umap_reduction.to_csv(fname, index=False)

    print('Uploading data to S3')
    upload(
        fname,
        os.path.join(S3_UMAP_PATH, fname), 
    )