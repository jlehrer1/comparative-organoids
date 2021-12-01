import pandas as pd 
import numpy as np
import hdbscan 
import pathlib 
import os 
import boto3 
import dask.dataframe as da 
import dask
from dask.diagnostics import ProgressBar
import sys
import argparse
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from helper import upload


def cluster(data):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=50)
    return clusterer.fit(data)

def generate_labels(N, COMP):
    here = pathlib.Path(__file__).parent.absolute()
    data_path = os.path.join(here, '..', '..', 'data', 'processed')

    print('Reading in organoid and primary data with Dask')
    primary = pd.read_csv(os.path.join(data_path, 'primary_reduction_neighbors_{N}_components_{COMP}.csv'))

    print('Computing primary clusters')
    prim_clusters = cluster(primary)

    print('Getting labels and writing to csv')
    prim_labels = np.array(prim_clusters.labels_)

    np.savetext('primary_labels_neighbors_{N}_components_{COMP}.csv', prim_labels, delimiter=',')

    print('Uploading to S3')
    upload(
        'primary_labels_neighbors_{N}_components_{COMP}.csv', 
        os.path.join('jlehrer', 'reduced_data', 'primary_labels_neighbors_{N}_components_{COMP}.csv')
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-N', type=int, required=False, default=100, help='Number of neighbors for UMAP data')
    parser.add_argument('-COMP', type=int, required=False, default=100, help='Number of components for UMAP data')
    args = parser.parse_args()

    N = args.N
    COMP = args.COMP
    
    generate_labels(N=N, COMP=COMP)