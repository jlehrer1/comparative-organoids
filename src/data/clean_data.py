import pandas as pd 
import numpy as np
import pathlib 
import os 

here = pathlib.Path(__file__).parent.absolute()
organoid = pd.read_csv(os.path.join(here, '..', '..', 'data', 'interim', 'organoid_T.tsv'), sep='\t')
primary = pd.read_csv(os.path.join(here, '..', '..', 'data', 'interim', 'primary_T.tsv'), sep='\t')

# Fix index name 
organoid.index = organoid.index.rename('cell')
primary.index = primary.index.rename('cell')

# Fix gene expression names in organoid data
organoid_cols = [x.split('|')[0] for x in organoid.columns]
organoid.columns = organoid_cols

# Consider only the genes between the two
subgenes = list(set(organoid.columns).intersection(primary.columns))

# Just keep those genes
organoid = organoid[subgenes]
primary = primary[subgenes]

# Fill NaN's with zeros
organoid = organoid.fillna(0)
primary = primary.fillna(0)

# Add type 
organoid['Type'] = [1]*organoid.shape[0] # 1 --> Organoid cell
primary['Type'] = [0]*primary.shape[0]

# Write to tsv 
organoid.to_csv(os.path.join(here, '..', '..', 'data', 'processed', 'organoid.tsv'), sep='\t', index=False)
primary.to_csv(os.path.join(here, '..', '..', 'data', 'processed', 'primary.tsv'), sep='\t', index=False)
