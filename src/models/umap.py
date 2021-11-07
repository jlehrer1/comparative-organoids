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

print('Reading in raw data')
df_organoid_raw = pd.read_csv(os.path.join(here, '..', '..', 'data', 'processed', 'organoid.tsv'), sep='\t')
df_primary_raw = pd.read_csv(os.path.join(here, '..', '..', 'data', 'processed', 'primary.tsv'), sep='\t')

comb = pd.concat([df_organoid_raw, df_primary_raw])

params = {
    'n_neighbors': [10, 50, 100, 1000, comb.shape[0]//4, comb.shape[0]//3],
    'min_dist': [0.1, 0.25, 0.5, 0.8, 0.99],
    'n_components': [2, 3],
}

print('Finding umap embeddings')
for neighbor in params['n_neighbors']:
    for dist in params['min_dist']:
        for n in params['n_components']:

            umap = draw_umap(
                n_neighbors=neighbor,
                min_dist=dist,
                n_components=n,
            )

            umap['Type'] = comb['Type'].apply(lambda x: 'Organoid' if x == 1 else 'Primary')

            umap.to_csv(f'comb_umap_nneigh_{neighbor}_mindist_{dist}_ncomp_{n}.tsv', sep='\t')

            print(f'Written UMAP with {neighbor} neighbors, {dist} distance, {n} components')
            print(f'Uploading UMAP with {neighbor} neighbors, {dist} distance, {n} components to S3')
            
            upload(f'comb_umap_nneigh_{neighbor}_mindist_{dist}_ncomp_{n}.tsv', f'comb_umap_nneigh_{neighbor}_mindist_{dist}_ncomp_{n}.tsv')