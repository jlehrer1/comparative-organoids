import boto3
import os 

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
        Key=os.path.join('jlehrer',  remote_name)
)

def download(remote_name, file_name=None):
    if file_name == None:
        file_name == remote_name

    s3.Bucket('braingeneersdev').download_file(
        Key=os.path.join('jlehrer', remote_name),
        Filename=file_name
    )

if __name__ == "__main__":
    pass