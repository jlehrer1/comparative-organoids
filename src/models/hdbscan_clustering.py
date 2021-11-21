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
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from helper import upload

@dask.delayed
def dask_cluster(data):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=50)
    return clusterer.fit(primary)

here = pathlib.Path(__file__).parent.absolute()
data_path = os.path.join(here, '..', '..', 'data', 'processed')

print('Reading in organoid and primary data with Dask')
organoid = da.read_csv(os.path.join(data_path, 'organoid.csv'))
primary = da.read_csv(os.path.join(data_path, 'primary.csv'))

print('Computing primary clusters')
prim_clusters = dask_cluster(organoid).compute()

print('Computing organoid clusters')
org_clusters = dask_cluster(primary).compute()

print('Getting labels and writing to csv')
prim_labels = np.array(prim_clusters.clusters_)
org_labels = np.array(org_clusters.clusters_)

np.savetext('primary_labels.csv', prim_labels, delimiter=',')
np.savetext('organoid_labels.csv', org_labels, delimiter=',')

print('Uploading to S3')
upload('primary_labels.csv')
upload('organoid_labels.csv')