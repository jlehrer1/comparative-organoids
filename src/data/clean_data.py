import pandas as pd 
import numpy as np
import pathlib 
import os 

here = pathlib.Path(__file__).parent.absolute()

print('Reading in raw organoid data')
organoid = pd.read_csv(os.path.join(here, '..', '..', 'data', 'interim', 'organoid_T.tsv'), sep='\t').set_index('gene', drop=True)

print('Reading in raw primary data')
primary = pd.read_csv(os.path.join(here, '..', '..', 'data', 'interim', 'primary_T.tsv'), sep='\t').set_index('gene', drop=True)

# Fix index name 
print('Setting indices')
organoid.index = organoid.index.rename('cell')
primary.index = primary.index.rename('cell')

# Fix gene expression names in organoid data
print('Fixing organoid column names')
organoid_cols = [x.split('|')[0] for x in organoid.columns]
organoid.columns = organoid_cols

# Consider only the genes between the two
print('Calculating gene intersection')
subgenes = list(set(organoid.columns).intersection(primary.columns))

# Just keep those genes
organoid = organoid[subgenes]
primary = primary[subgenes]

# Fill NaN's with zeros
print('Filling NaN\'s with zeros')
organoid = organoid.fillna(0)
primary = primary.fillna(0)

print('Removing all zero columns in organoid and primary data')
for col in subgenes:
    if (organoid[col] == 0).all():
        organoid = organoid.drop(col, axis=1)

    if (primary[col] == 0).all():
        primary = primary.drop(col, axis=1)

# Add type
print('Adding type column')
organoid['Type'] = [1]*organoid.shape[0] # 1 --> Organoid cell
primary['Type'] = [0]*primary.shape[0]

# Write to tsv 
print('Writing out clean organoid data to tsv')
organoid.to_csv(os.path.join(here, '..', '..', 'data', 'processed', 'organoid.tsv'), sep='\t')

print('Writing out clean primary data to tsv')
primary.to_csv(os.path.join(here, '..', '..', 'data', 'processed', 'primary.tsv'), sep='\t')
