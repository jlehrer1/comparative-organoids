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
   
def upload(file_name, remote_name):
    s3.Bucket('braingeneersdev').upload_file(
        Filename=file_name,
        Key=os.path.join('jlehrer', 'mo_data', remote_name)
    )

here = pathlib.Path(__file__).parent.absolute()

print('Reading in raw data using Dask')
df_organoid_raw = dd.read_csv(os.path.join(here, '..', '..', 'data', 'raw', 'organoid.tsv'), sep='\t', sample=10000000)
df_primary_raw = dd.read_csv(os.path.join(here, '..', '..', 'data', 'raw', 'primary.tsv'), sep='\t', sample=10000000)

# print('Fixing index on organoid data')
# df_organoid_raw['gene'].apply(lambda x: x.split('|')[0]) # For some reason gene expressions are marked twice so just fix this quickly

print('Dropping gene column so everything is numeric')
df_organoid_raw = df_organoid_raw.drop('gene', axis=1)
df_primary_raw = df_primary_raw.drop('gene', axis=1)

print('Casting data to array so it can be transposed')
df_primary_raw = df_primary_raw.values
df_primary_raw = df_organoid_raw.values

print('Transposing array')
df_primary_raw = df_primary_raw.T
df_organoid_raw = df_organoid_raw.T

print('Casting back to csv')
df_primary_raw = dd.from_dask_array(df_primary_raw)
df_organoid_raw = dd.from_dask_array(df_organoid_raw)

print('Writing out to csv')
df_primary_raw.to_csv('df_primary_raw.csv', index=False)
df_organoid_raw.to_csv('df_organoid_raw.csv', index=False)

print('Uploading to s3')
upload('df_primary_raw_tranposed.csv', 'df_primary_raw_tranposed.csv')
upload('df_organoid_raw.csv', 'df_organoid_raw.csv')
