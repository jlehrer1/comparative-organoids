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
   
def upload(file_name, remote_name=None) -> None:
    """
    Uploads a file to the braingeneersdev S3 bucket
    
    Parameters:
    file_name: Local file to upload
    remote_name: Key for S3 bucket. Default is file_name
    """
    if remote_name == None:
        remote_name = file_name

    s3.Bucket('braingeneersdev').upload_file(
        Filename=file_name,
        Key=remote_name,
)

def download(remote_name, file_name=None) -> None:
    """
    Downloads a file from the braingeneersdev S3 bucket 

    Parameters:
    remote_name: S3 key to download. Must be a single file
    file_name: File name to download to. Default is remote_name
    """
    if file_name == None:
        file_name == remote_name

    s3.Bucket('braingeneersdev').download_file(
        Key=remote_name,
        Filename=file_name,
    )

def umap_plot(data, title=None) -> None:
    """
    Generates the scatterplot of clustered 2d dimensional data, where the cluster name column is 'label'.

    Parameters:
    data: n x 3 DataFrame, where one of the columns are the cluster labels and the other two are the UMAP dimensions
    title: Optional title to append to the UMAP plot 
    """
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

def list_objects(prefix: str) -> list:
    """
    Lists all the S3 objects from the braingeneers bucket with the given prefix.

    Parameters:
    prefix: Prefix to filter S3 objects. For example, if we want to list all the objects under 'jlehrer/data' we pass 
            prefix='jlehrer/data'

    Returns:
    list[str]: List of objects with given prefix
    """
    objs = s3.Bucket('braingeneersdev').objects.filter(Prefix=prefix)
    return [x.key for x in objs]

if __name__ == "__main__":
    pass