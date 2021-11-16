import boto3 
import pathlib 
import os 

s3 = boto3.resource(
    's3',
    endpoint_url="https://s3.nautilus.optiputer.net",
    aws_access_key_id="EFIE1S59OR5CHDC4KCHK",
    aws_secret_access_key="DRXgeKsTLctfFX9udqfT04go8JpxG3qWxj0OKHVU",
)
   
def download(remote_name, file_name=None):
    if file_name == None:
        file_name == remote_name

    s3.Bucket('braingeneersdev').download_file(
        Key=os.path.join('jlehrer', 'transposed_data', 'clean', '', remote_name),
        Filename=file_name
    )

here = pathlib.Path(__file__).parent.absolute()

print('Downloading clean organoid data from S3')
download('organoid.csv', os.path.join(here, '..', '..', 'data', 'interim', 'organoid_T.csv'))

print('Downloading raw primary data from S3')
download('primary_T.csv', os.path.join(here, '..', '..', 'data', 'interim', 'primary_T.csv'))