import dask.dataframe as dd
from dask_ml.decomposition import PCA
import os, sys
import argparse
import pathlib 
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from helper import upload


def pca(data, n_components):
    est = PCA(n_components=n_components)
    return est.fit_transform(data.values.compute_chunk_sizes())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage='Compute PCA reduction for organoid and primary data.')

    parser.add_argument(
        '--file',
        help='Which of the files to compute the PCA on',
        required=True,
        type=str,
        choices=['organoid', 'primary']
    )

    parser.add_argument(
        '--n',
        help='Number of components to keep after calculating PCA',
        required=True,
        type=int
    )

    here = pathlib.Path(__file__).parent.absolute()

    args = parser.parse_args()
    n_components = args.n 

    organoid = dd.read_csv(os.p)