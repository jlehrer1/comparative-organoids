import os
import pathlib 
import sys

import pandas as pd 

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from helper import primary_genes, upload

here = pathlib.Path(__file__).parent.absolute()
path = os.path.join(here, '..', '..', 'data', 'processed')
files = [f.rstrip() for f in os.listdir(path) if f.startswith('primary_labels')]
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
        print(f'Calculating for cluster {clust}')
        rows = labels[labels['# label'] == clust].index
        try:
            df = pd.read_csv(os.path.join(path, 'primary.csv'), skiprows = lambda x: x not in rows, names=cols)
            df.columns = [c.lower() for c in df.columns]
            df = df.drop(to_remove, axis=1)
            cluster_df.loc[:, clust] = df.sum().nlargest(1000).index
        except Exception as e:
            print(f'Error with {label_file} in cluster {clust}, continuing...')
            print(str(e))
        print(cluster_df)
        
    print(f'Saving and uploading top genes for cluster run')
    cluster_df.to_csv(f'annotation_{label_file}')
    upload(
        file_name=f'annotation_{label_file}', 
        remote_name=f'jlehrer/cluster_annotation/annotation_{label_file}'
    )
