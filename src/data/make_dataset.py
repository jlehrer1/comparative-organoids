import pandas as pd 
import numpy as np
import pathlib 
import os 
import umap
import dask.dataframe as dd

def draw_umap(data, n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean'):
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric
    )
    return fit.fit_transform(data);
    
here = pathlib.Path(__file__).parent.absolute()

print('Reading in raw data using Dask')
df_organoid_raw = dd.read_csv(os.path.join(here, '..', '..', 'data', 'raw', 'organoid.tsv.gz'), compression='gzip', sep='\t')
df_primary_raw = dd.read_csv(os.path.join(here, '..', '..', 'data', 'raw', 'primary.tsv.gz'), compression='gzip', sep='\t')

print('Setting index and transposing data')
df_organoid_raw['gene'].apply(lambda x: x.split('|')[0]) # For some reason gene expressions are marked twice so just fix this quickly

df_primary_raw = df_primary_raw.set_index('gene')
df_organoid_raw = df_organoid_raw.set_index('gene')

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

            df_primary.to_csv(f'primary_neighbors_{neighbor}_dist_{dist}_components_{n}.csv', index=False)
            df_organoid.to_csv(f'organoid_neighbors_{neighbor}_dist_{dist}_components_{n}.csv', index=False)

