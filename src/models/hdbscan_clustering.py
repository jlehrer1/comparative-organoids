import pandas as pd 
import numpy as np
import hdbscan 
import pathlib 
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
        Key=os.path.join('jlehrer', 'mo_data', remote_name)
    )

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
