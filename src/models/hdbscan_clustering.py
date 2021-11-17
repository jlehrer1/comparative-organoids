import pandas as pd 
import numpy as np
import hdbscan 
import pathlib 
import os 
import boto3 
import dask.dataframe as da 
import dask
from dask.diagnostics import ProgressBar

s3 = boto3.resource(
    's3',
    endpoint_url="https://s3.nautilus.optiputer.net",
    aws_access_key_id="EFIE1S59OR5CHDC4KCHK",
    aws_secret_access_key="DRXgeKsTLctfFX9udqfT04go8JpxG3qWxj0OKHVU",
)

def upload(file_name, remote_name=None):
    if remote_name == None:
        remote_name = file_name

    s3.Bucket('braingeneersdev').upload_file(
        Filename=file_name,
        Key=os.path.join('jlehrer', 'mo_data', remote_name)
    )

@dask.delayed
def dask_cluster(data):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=3)
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


