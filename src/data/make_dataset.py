import pandas as pd 
import numpy as np
import pathlib 
import os 
import umap
import dask.dataframe as dd

here = pathlib.Path(__file__).parent.absolute()

df_organoid_raw = pd.read_csv(os.path.join(here, '..', '..', 'data', 'raw', 'organoid.tsv.gz'))
df_primary_raw = pd.read_csv(os.path.join(here, '..', '..', 'data', 'raw', 'primary.tsv.gz'))

df_organoid_raw['gene'].apply(lambda x: x.split('|')[0]) # For some reason gene expressions are marked twice so just fix this quickly
df_primary_raw = df_primary_raw.set_index('gene')
df_organoid_raw = df_organoid_raw.set_index('gene')

df = dd.read_csv('Data/primary.tsv.gz', compression='gzip', sep='\t', sample=1000000000000, blocksize=None)