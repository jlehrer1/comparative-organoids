import pandas as pd 
import numpy as np
import pathlib 
from tqdm import tqdm
import os 
import boto3 

s3 = boto3.resource(
    's3',
    endpoint_url="https://s3.nautilus.optiputer.net",
    aws_access_key_id="EFIE1S59OR5CHDC4KCHK",
    aws_secret_access_key="DRXgeKsTLctfFX9udqfT04go8JpxG3qWxj0OKHVU",
)
   
def upload(file_name, remote_name=None):
    if remote_name == None:
        remote_name = file_name

    s3.Bucket('braingeneersdev').upload_file(
        Filename=file_name,
        Key=os.path.join('jlehrer', 'transposed_data', 'clean', remote_name)
    )

def download(remote_name, file_name=None):
    if file_name == None:
        file_name == remote_name

    s3.Bucket('braingeneersdev').download_file(
        Key=os.path.join('jlehrer', 'transposed_data', remote_name),
        Filename=file_name
    )

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
# Maybe remove this once we have the full transposed dataset
for col in tqdm(subgenes):
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

print('Uploading files to S3')
upload(os.path.join(here, '..', '..', 'data', 'processed', 'primary.tsv'))
upload(os.path.join(here, '..', '..', 'data', 'processed', 'organoid.tsv'))
