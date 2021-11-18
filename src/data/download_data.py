import boto3 
import pathlib 
import os 
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from helper import download

def download_all():
    here = pathlib.Path(__file__).parent.absolute()

    if not os.path.isfile(os.path.join(here, '..', '..', 'data', 'clean', 'organoid.csv')):
        print('Downloading clean organoid data from S3')
        download(
            os.path.join('transposed_data', 'clean', 'organoid.csv'), 
            os.path.join(here, '..', '..', 'data', 'processed', 'organoid.csv')
        )

    if not os.path.isfile(os.path.join(here, '..', '..', 'data', 'clean', 'organoid.csv')):
        print('Downloading raw primary data from S3')

        download(
            os.path.join('transposed_data', 'clean', 'primary.csv'), 
            os.path.join(here, '..', '..', 'data', 'processed', 'primary.csv')
        )

if __name__ == "__main__":
    download_all()