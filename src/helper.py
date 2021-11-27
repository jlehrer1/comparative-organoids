import boto3
import os 
import matplotlib.pyplot as plt 
import seaborn as sns 

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

def umap_plot(data, title):
    fig, ax = plt.subplots(figsize=(15, 10))

    sns.scatterplot(
        x='0', 
        y='1',
        data=data,
        hue='label',
        legend='full',
        ax=ax,
        s=1,
        palette='bright'
    )

    plt.title(f'UMAP Projection: {title}')
    plt.savefig(f'umap_cluster_{title}.png', dpi=300)

if __name__ == "__main__":
    pass