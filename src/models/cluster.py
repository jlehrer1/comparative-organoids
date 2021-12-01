import pandas as pd 
import numpy as np
import hdbscan 
import pathlib 
import os 
import dask.dataframe as da 
import sys
import argparse
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from helper import upload, S3_CLUSTER_LABEL_PATH

def cluster(data, min_cluster_size):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, core_dist_n_jobs=1)
    return clusterer.fit(data)

def generate_labels(N, COMP, min_cluster_size):
    here = pathlib.Path(__file__).parent.absolute()
    data_path = os.path.join(here, '..', '..', 'data', 'processed')
    fname = f'primary_labels_neighbors_{N}_components_{COMP}_clust_size_{min_cluster_size}.csv'

    print('Reading in primary data with Dask')
    primary = pd.read_csv(os.path.join(data_path, f'primary_reduction_neighbors_{N}_components_{COMP}.csv'))

    print('Computing primary clusters')
    prim_clusters = cluster(primary, min_cluster_size)

    print('Getting labels and writing to csv')
    prim_labels = np.array(prim_clusters.labels_)

    np.savetxt(fname, prim_labels, delimiter=',', header='label')

    print('Uploading to S3')
    upload(
        fname,
        os.path.join(S3_CLUSTER_LABEL_PATH, fname)
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-N', type=int, required=False, default=500, help='Number of neighbors for UMAP data')
    parser.add_argument('-COMP', type=int, required=False, default=100, help='Number of components for UMAP data')
    parser.add_argument('-M', type=int, required=False, default=250, help='Min cluster size for HDBSCAN')
    args = parser.parse_args()

    N = args.N
    COMP = args.COMP
    MIN_CLUST_SIZE = args.M

    generate_labels(
        N=N, 
        COMP=COMP, 
        min_cluster_size=MIN_CLUST_SIZE
    )