import pandas as pd 
import numpy as np
import pathlib 
import os 
import umap
import dask.dataframe as dd
import boto3
    
s3 = boto3.resource(
    's3',
    endpoint_url="https://s3.nautilus.optiputer.net",
    aws_access_key_id="EFIE1S59OR5CHDC4KCHK",
    aws_secret_access_key="DRXgeKsTLctfFX9udqfT04go8JpxG3qWxj0OKHVU",
)

def draw_umap(data, n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean'):
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric
    )
    return fit.fit_transform(data);
    
def upload(file_name, remote_name):
    s3.Bucket('braingeneersdev').upload_file(
        Filename=file_name,
        Key=os.path.join('jlehrer', 'mo_data', remote_name)
    )

here = pathlib.Path(__file__).parent.absolute()

print('Reading in raw data using Dask')
df_organoid_raw = dd.read_csv(os.path.join(here, '..', '..', 'data', 'raw', 'organoid.tsv'), sep='\t', sample=10000000)
df_primary_raw = dd.read_csv(os.path.join(here, '..', '..', 'data', 'raw', 'primary.tsv'), sep='\t', sample=10000000)

print('Fixing index on organoid data')
df_organoid_raw['gene'].apply(lambda x: x.split('|')[0]) # For some reason gene expressions are marked twice so just fix this quickly

print('Setting index')
df_primary_raw = df_primary_raw.set_index('gene')
df_organoid_raw = df_organoid_raw.set_index('gene')

print('Transposing data')
df_primary_raw = df_primary_raw.T
df_organoid_raw = df_organoid_raw.T

params = {
    'n_neighbors': [10, 50, 100, 1000, 10000, 50000, 100000],
    'min_dist': [0.1, 0.25, 0.5, 0.8, 0.99],
    'n_components': [2, 3],
}

print('Finding umap embeddings')
for neighbor in params['n_neighbors']:
    for dist in params['min_dist']:
        for n in params['n_components']:
            df_primary = draw_umap(
                data=df_primary_raw,
                n_neighbors=neighbor,
                min_dist=dist,
                n_components=n,
            )

            df_organoid = draw_umap(
                data=df_organoid_raw,
                n_neighbors=neighbor,
                min_dist=dist,
                n_components=n,
            )

            df_primary.to_csv(
                os.path.join(here, '..', '..', 'data', 'interim', f'primary_neighbors_{neighbor}_dist_{dist}_components_{n}.csv'), 
                index=False
            )

            df_organoid.to_csv(
                os.path.join(here, '..', '..', 'data', 'interim', f'organoid_neighbors_{neighbor}_dist_{dist}_components_{n}.csv'), 
                index=False
            )

            print(f'Written UMAP with {neighbor} neighbors, {dist} distance, {n} components')
            print(f'Uploading UMAP with {neighbor} neighbors, {dist} distance, {n} components to S3')
            
            upload(
                os.path.join(here, '..', '..', 'data', 'interim', f'primary_neighbors_{neighbor}_dist_{dist}_components_{n}.csv'),
                f'primary_neighbors_{neighbor}_dist_{dist}_components_{n}.csv'
            )

            upload(
                os.path.join(here, '..', '..', 'data', 'interim', f'organoid_neighbors_{neighbor}_dist_{dist}_components_{n}.csv'),
                f'organoid_neighbors_{neighbor}_dist_{dist}_components_{n}.csv'
            )