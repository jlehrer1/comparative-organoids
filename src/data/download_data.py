import pathlib 
import os 
import sys
import argparse
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from helper import download, list_objects

here = pathlib.Path(__file__).parent.absolute()
data_path = os.path.join(here, '..', '..', 'data')

def download_clean(type=None) -> None:
    """
    Downloads the cleaned organoid and primary cell dataset
    """
    if type == 'organoid':
        print(f'Downloading organoid from S3')
        download(
            os.path.join('jlehrer', 'organoid.csv'), 
            os.path.join(data_path, 'processed', 'organoid.csv')
        )
    elif type == 'primary':
        print(f'Downloading primary from S3')
        download(
            os.path.join('jlehrer', 'primary.csv'), 
            os.path.join(data_path, 'processed', 'primary.csv')
        )
    else:
        for f in 'organoid.csv', 'primary.csv':
            if not os.path.isfile(os.path.join(data_path, 'processed', f)):
                print(f'Downloading {f} from S3')
                download(
                    os.path.join('jlehrer', f), 
                    os.path.join(data_path, 'processed', f)
                )

def download_reduced() -> None:
    """
    Downloads all the UMAP projections of the primary and organoid data from the braingeneersdev S3 bucket
    """
    reduced_files = list_objects(os.path.join('jlehrer', 'reduced_data'))

    for f in reduced_files:
        if not os.path.isfile(os.path.join(data_path, 'processed', f.split('/')[-1])):
            print(f'Downloading {f} from S3')
            download(
                f,
                os.path.join(data_path, 'processed', f.split('/')[-1]) # Just the file name in the list of objects
            )

def download_labels() -> None:
    """
    Downloads the cluster labels for primary data
    """

    label_files = list_objects(os.path.join('jlehrer', 'primary_cluster_labels'))

    for f in label_files:
        if not os.path.isfile(os.path.join(data_path, 'processed', f.split('/')[-1])):
            download(
                f,
                os.path.join(data_path, 'processed', f.split('/')[-1]) # Just the file name in the list of objects
            )
    
def download_raw():
    raise NotImplementedError()

def download_interim() -> None:
    """Downloads the interim data from S3. Interim data is in the correct structural format but has not been cleaned."""

    for f in 'organoid_T.csv', 'primary_T.csv':
        if not os.path.isfile(os.path.join(data_path, 'interim', f)):
            print(f'Downloading {f} data from S3')
            download(
                os.path.join('jlehrer', 'transposed_data', f), 
                os.path.join(data_path, 'interim', f)
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-type',
        type=str,
        required=False,
        default='clean',
        choices=['clean', 'interim', 'raw', 'reduced', 'labels'],
        help="Type of data to download. Can be one of ['clean', 'interim', 'raw', 'reduced', 'labels']"
    )
    args = parser.parse_args()
    type = args.type

    if type == 'clean':
        download_clean()
    elif type == 'interim':
        download_interim()
    elif type == 'raw':
        download_raw()
    elif type == 'labels':
        download_labels()
    else:
        download_reduced()