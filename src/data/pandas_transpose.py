import pandas as pd
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

print('Reading in raw organoid tsv')
organoid = pd.read_csv(os.path.join(here, '..', '..', 'data', 'raw', 'organoid.tsv'), sep='\t')

print('Reading in raw primary tsv')
primary = pd.read_csv(os.path.join(here, '..', '..', 'data', 'raw', 'primary.tsv'), sep='\t')

print('Transposing organoid tsv')
organoid = organoid.T

print('Transposing primary tsv')
primary = primary.T

print('Writing out organoid tsv')
organoid.to_csv('organoid_T_pandas.tsv', sep='\t')

print('Writing out primary tsv')
primary.to_csv('primary_T_pandas.tsv', sep='\t')

print('Uploading organoid data')
upload('organoid_T_pandas.tsv')

print('Uploading primary data')
upload('primary_T_pandas.tsv')


