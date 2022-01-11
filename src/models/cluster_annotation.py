import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import os
import dask.dataframe as dd
import pathlib 
import dask.dataframe as dd

from helper import primary_genes
from helper import upload

here = pathlib.Path(__file__).parent.absolute()
path = os.path.join(here, '..', '..', 'processed')
files = [f.rstrip for f in os.listdir(os.path.join(path, 'labels'))]
cols = primary_genes()

# Generate the list of mitochondrial and ribosomal genes to remove 
to_remove = []
for c in cols:
    if c.startswith('rp') or c.startswith('mt'):
        to_remove.append(c)

for label_file in files:
    print(f'Calculating top genes for cluster run {label_file}')
    labels = pd.read_csv(os.path.join(path, label_file))
    cluster_df = pd.DataFrame()

    for clust in labels.value_counts().index:
        rows = labels[labels['# label'] == clust].index
        df = pd.read_csv('../data/processed/primary.csv', skiprows = lambda x: x not in rows, names=cols)
        df.columns = [c.lower() for c in df.columns]
        df = df.drop(to_remove, axis=1)

        cluster_df.loc[:, clust] = df.sum().nlargest(1000).index
    
    print(f'Saving and uploading top genes for cluster run')
    cluster_df.to_csv(f'annotation_{label_file}')
    upload(
        file_name=f'annotation_{label_file}', 
        remote_name=f'jlehrer/cluster_annotation/annotation_{label_file}'
    )
