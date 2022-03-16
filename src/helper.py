import boto3
import os 
import matplotlib.pyplot as plt 
import seaborn as sns 
import pathlib 
import json
import pandas as pd 
from typing import * 

S3_CLUSTER_LABEL_PATH = os.path.join('jlehrer', 'primary_cluster_labels')
S3_CLEAN_DATA_PATH = 'jlehrer'
S3_UMAP_PATH = os.path.join('jlehrer', 'reduced_data')

DATA_FILES_AND_URLS_DICT = {
    'primary_bhaduri.tsv': [
        'https://cells.ucsc.edu/organoidreportcard/primary10X/exprMatrix.tsv.gz', 
        'https://cells.ucsc.edu/organoidreportcard/primary10X/meta.tsv',
    ],
    'allen_cortex.tsv': [
        'https://cells.ucsc.edu/allen-celltypes/human-cortex/various-cortical-areas/exprMatrix.tsv.gz',
        'https://cells.ucsc.edu/allen-celltypes/human-cortex/various-cortical-areas/meta.tsv',
    ],
    'allen_m1_region.tsv': [
        'https://cells.ucsc.edu/allen-celltypes/human-cortex/m1/exprMatrix.tsv.gz',
        'https://cells.ucsc.edu/allen-celltypes/human-cortex/m1/meta.tsv',
    ],
    'whole_brain_bhaduri.tsv': [
        'https://cells.ucsc.edu/dev-brain-regions/wholebrain/exprMatrix.tsv.gz',
        'https://cells.ucsc.edu/dev-brain-regions/wholebrain/meta.tsv',
    ],
}

DATA_FILES_AND_NAMES_DICT = {
    'primary_bhaduri.tsv': 'Bhaduri et. al (2019)',
    'allen_cortex.tsv': 'Allen Brain Atlas Cortex',
    'allen_m1_region.tsv': 'Allen Brain Atlas M1 Region',
    'whole_brain_bhaduri.tsv': 'Bhaduri et. al (2021)'
}


INTERIM_DATA_AND_LABEL_FILES_LIST = {
    'primary_bhaduri_T.csv': 'primary_bhaduri_labels.csv',
    'allen_cortex_T.csv': 'allen_cortex_labels.csv',
    'allen_m1_region_T.csv': 'allen_m1_region_labels.csv',
    'whole_brain_bhaduri_T.csv': 'whole_brain_bhaduri_labels.csv'
}

DATA_FILES_LIST = DATA_FILES_AND_URLS_DICT.keys()
DATA_URLS_LIST = DATA_FILES_AND_URLS_DICT.values()

here = pathlib.Path(__file__).parent.absolute()

with open(os.path.join(here, '..', 'credentials')) as f:
    key, access = [line.rstrip() for line in f.readlines()]

s3 = boto3.resource(
    's3',
    endpoint_url="https://s3.nautilus.optiputer.net",
    aws_access_key_id=key,
    aws_secret_access_key=access,
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
    List[str]: List of objects with given prefix
    """

    objs = s3.Bucket('braingeneersdev').objects.filter(Prefix=prefix)
    return [x.key for x in objs]

def fix_labels(
    file: str, 
    path: str, 
    class_label: str='# label',
):
    """
    Fixes label output from HDBSCAN to be non-negative, since noise points are classified with label -1. PyTorch requires indexing from 0. 

    Parameters:
    file: Path to label file
    path: Path to write corrected label file to
    """

    labels = pd.read_csv(file)
    labels[class_label] = labels[class_label].astype(int) + 1
    labels.to_csv(os.path.join(path, 'fixed_' + file.split('/')[-1]))

def primary_genes() -> list:
    """Return a list of all primary genes excluding mitochondrial and ribosomal genes"""

    with open(os.path.join(here, 'genes.json'), 'r') as f:
        arr = json.load(f)

    return arr

def gene_intersection() -> List[str]:
    files = DATA_FILES_LIST
    files = [f'{file[:-4]}_T.csv' for file in files]

    cols = []
    for file in files:
        # Read in columns, split by | (since some are PVALB|PVALB), and make sure all are uppercase
        temp = pd.read_csv(os.path.join(here, '..', 'data', 'interim', file), nrows=1, header=1).columns 
        temp = [x.split('|')[0].upper() for x in temp]
        cols.append(set(temp))

    unique = list(set.intersection(*cols))
    unique = sorted(unique)

    return unique 