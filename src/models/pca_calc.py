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
    data_path = os.path.join(here, '..', '..', 'data', 'processed')

    args = parser.parse_args()
    COMP = args.n 
    FILE = args.file

    data = dd.read_csv(os.path.join(data_path, f'{FILE}.csv'), assume_missing=True)

    pca_data = pca(data, COMP)

    fname = f'pca_components_{COMP}_{FILE}.csv'
    pca_data.to_csv(fname, single_file=True, index=False)
    
    upload(
        fname,
        os.path.join('jlehrer', 'pca_data', fname)
    )