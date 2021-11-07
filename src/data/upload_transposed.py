import pandas as pd 
import numpy as np
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

# upload(os.path.join(here, '..', '..', 'organoid_T_inmem.tsv'))
upload(os.path.join(here, '..', '..', 'primary_T.tsv'))