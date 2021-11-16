import pandas as pd 
import numpy as np
import pathlib 
import os 
import umap
import dask.dataframe as dd
import boto3
import matplotlib.pyplot as plt 
import seaborn as sns
import itertools 
import dask.dataframe as da

s3 = boto3.resource(
    's3',
    endpoint_url="https://s3.nautilus.optiputer.net",
    aws_access_key_id="EFIE1S59OR5CHDC4KCHK",
    aws_secret_access_key="DRXgeKsTLctfFX9udqfT04go8JpxG3qWxj0OKHVU",
)

def upload(file_name, remote_name):
    if remote_name == None:
        remote_name = file_name

    s3.Bucket('braingeneersdev').upload_file(
        Filename=file_name,
        Key=os.path.join('jlehrer', 'mo_data', remote_name)
    )

def plot_umap(umap_data, title):
    fig, ax = plt.subplots(figsize=(10, 10))

    sns.scatterplot(
        x="UMAP_1", 
        y="UMAP_2", 
        data=umap_data,
        hue='Type',
        legend='full',
        ax=ax,
    )

    plt.title(title)
    fig.savefig(f'{title}.png')

def umap_calc(data, n_neighbors, min_dist):
    fit = umap.UMAP(
        n_neighbors=neighbor,
        min_dist=dist,
        verbose=True,
        random_state=42,
    )

    umap_df = fit.fit_transform(data.drop('Type', axis=1));
    umap_df = umap_df.rename({0: 'UMAP_1', 1:'UMAP_2'}, axis=1)

    return umap_df


here = pathlib.Path(__file__).parent.absolute()

print('Reading in organoid data')
organoid = da.read_csv(os.path.join(here, '..', '..', 'data', 'processed', 'organoid.csv'))

print('Reading in primary data')
primary = da.read_csv(os.path.join(here, '..', '..', 'data', 'processed', 'primary.csv'))

print('Joining DataFrames')
comb = pd.concat([organoid, primary])
comb = comb.compute()

params = {
    'n_neighbors': [15, 100, 1000, comb.shape[0]//4, comb.shape[0]//3],
    'min_dist': [0.1, 0.25, 0.5, 0.8, 0.99],
    'n_components': [2, 3],
    'metric': ['euclidean', 'manhattan', 'mahalanobis'],
}

print('Finding umap embeddings')
for neighbor, dist in itertools.product(params['n_neighbors'], params['min_dist']):
    print(f'Calculating UMAP with {neighbor} neighbors, {dist} min dist')

    comb_df = umap_calc(comb, neighbor, dist).compute()
    org_df = umap_calc(organoid, neighbor, dist).compute()
    prim_df = umap_calc(organoid, neighbor, dist).compute()

    # Generate UMAP plot
    plot_umap(comb_df,f'comb_umap_nneigh_{neighbor}_{dist}')
    plot_umap(org_df,f'org_umap_nneigh_{neighbor}_{dist}')
    plot_umap(prim_df,f'prim_umap_nneigh_{neighbor}_{dist}')


    print('Writing UMAP data to csv')
    comb_df.to_csv(f'comb_umap_nneigh_{neighbor}_{dist}.csv')
    org_df.to_csv(f'org_umap_nneigh_{neighbor}_{dist}.csv')
    prim_df.to_csv(f'prim_umap_nneigh_{neighbor}_{dist}.csv')

    # Upload data and plots
    print('Uploading combined data to S3')
    upload(f'comb_umap_nneigh_{neighbor}_{dist}.csv')
    upload(f'comb_umap_nneigh_{neighbor}_{dist}.png')

    print('Uploading organoid data to S3')
    upload(f'org_umap_nneigh_{neighbor}_{dist}.csv')
    upload(f'org_umap_nneigh_{neighbor}_{dist}.png')

    print('Uploading primary data to S3')
    upload(f'prim_umap_nneigh_{neighbor}_{dist}.csv')
    upload(f'prim_umap_nneigh_{neighbor}_{dist}.png')

