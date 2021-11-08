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

s3 = boto3.resource(
    's3',
    endpoint_url="https://s3.nautilus.optiputer.net",
    aws_access_key_id="EFIE1S59OR5CHDC4KCHK",
    aws_secret_access_key="DRXgeKsTLctfFX9udqfT04go8JpxG3qWxj0OKHVU",
)

def upload(file_name, remote_name):
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
        markers=["+", "-"],
        ax=ax,
    )

    plt.title(title)
    fig.savefig(f'{title}.png')

here = pathlib.Path(__file__).parent.absolute()

print('Reading in organoid data')
df_organoid_raw = (pd.read_csv(
    os.path.join(here, '..', '..', 'data', 'processed', 'organoid.tsv'), sep='\t')
    .set_index('cell', drop=True)
)

print('Reading in primary data')
df_primary_raw = (pd.read_csv(
    os.path.join(here, '..', '..', 'data', 'processed', 'primary.tsv'), sep='\t')
    .set_index('cell', drop=True)
)

print('Joining DataFrames')
comb = pd.concat([df_organoid_raw, df_primary_raw])

params = {
    'n_neighbors': [15, 100, 1000, comb.shape[0]//4, comb.shape[0]//3],
    'min_dist': [0.1, 0.25, 0.5, 0.8, 0.99],
    'n_components': [2, 3],
    'metric': ['euclidean', 'manhattan', 'mahalanobis', ],
}

print('Finding umap embeddings')
for neighbor, dist, metric in itertools.product(params['n_neighbors'], params['min_dist'], params['metric']):
    print(f'Calculating UMAP with {neighbor} neighbors, {dist} min dist, metric {metric}')

    fit = umap.UMAP(
        n_neighbors=neighbor,
        min_dist=dist,
        metric=metric,
        verbose=True,
    )

    umap_df = fit.fit_transform(comb.drop('Type', axis=1));

    umap_df = pd.DataFrame(umap_df, index=comb.index)
    umap_df['Type'] = comb['Type'].apply(lambda x: 'Organoid' if x == 1 else 'Primary')
    umap_df = umap_df.rename({0: 'UMAP_1', 1:'UMAP_2'}, axis=1)

    # Generate UMAP plot
    plot_umap(
        umap_df,
        f'comb_umap_nneigh_{neighbor}_{dist}_{metric}'
    )

    # Write data to csv
    print('Writing UMAP data to csv')
    umap_df.to_csv(f'comb_umap_nneigh_{neighbor}_{dist}_{metric}.tsv', sep='\t')

    # Upload data and plots
    print('Uploading UMAP data to S3')
    upload(f'comb_umap_nneigh_{neighbor}_{dist}_{metric}.tsv', f'comb_umap_nneigh_{neighbor}_{dist}_{metric}.tsv')
    upload(f'comb_umap_nneigh_{neighbor}_{dist}_{metric}.png', f'comb_umap_nneigh_{neighbor}_{dist}_{metric}.png')