import pathlib 
import os 
import sys
import argparse
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from helper import download, list_objects

here = pathlib.Path(__file__).parent.absolute()
data_path = os.path.join(here, '..', '..', 'data')

def download_clean():
    """
    Downloads the cleaned organoid and primary cell dataset
    """
    if not os.path.isfile(os.path.join(data_path, 'organoid.csv')):
        print('Downloading clean organoid data from S3')
        download(
            os.path.join('jlehrer', 'organoid.csv'), 
            os.path.join(data_path, 'processed', 'organoid.csv')
        )

    if not os.path.isfile(os.path.join(data_path, 'primary.csv')):
        print('Downloading raw primary data from S3')
        download(
            os.path.join('jlehrer', 'primary.csv'), 
            os.path.join(data_path, 'processed', 'primary.csv')
        )

def download_reduced():
    reduced_files = list_objects(os.path.join('jlehrer', 'reduced_data'))

    for f in reduced_files:
        download(
            f,
            os.path.join(data_path, 'processed', f.split('/')[-1])
        )

def download_raw():
    pass

def download_interim():
    if not os.path.isfile(os.path.join(data_path, 'organoid_T.csv')):
        print('Downloading interim organoid data from S3')

        download(
            os.path.join('jlehrer', 'transposed_data', 'organoid_T.csv'), 
            os.path.join(data_path, 'interim', 'organoid_T.csv')
        )

    if not os.path.isfile(os.path.join(data_path, 'primary_T.csv')):
        print('Downloading interim primary data from S3')

        download(
            os.path.join('jlehrer', 'transposed_data', 'primary_T.csv'), 
            os.path.join(data_path, 'interim', 'primary_T.csv')
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-type',
        type=str,
        required=False,
        default='clean',
        help="Type of data to download. Can be one of ['clean', 'interim', 'raw', 'reduced']"
    )
    args = parser.parse_args()
    type = args.type

    if type == 'clean':
        download_clean()
    elif type == 'interim':
        download_interim()
    elif type == 'raw':
        download_raw()
    elif type == 'reduced':
        download_reduced()
    else:
        raise ValueError("Invalid value for type. Can be one of ['clean', 'interim', 'raw', 'reduced']")