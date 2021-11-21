import pathlib 
import os 
import sys
import argparse
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from helper import download

here = pathlib.Path(__file__).parent.absolute()
data_path = os.path.join(here, '..', '..', 'data', 'processed')

def download_clean():
    if not os.path.isfile(os.path.join(data_path, 'organoid.csv')):
        print('Downloading clean organoid data from S3')

        download(
            os.path.join('organoid.csv'), 
            os.path.join(data_path, 'organoid.csv')
        )

    if not os.path.isfile(os.path.join(data_path, 'primary.csv')):
        print('Downloading raw primary data from S3')

        download(
            os.path.join('primary.csv'), 
            os.path.join(data_path, 'primary.csv')
        )

def download_reduced():
    pass

if __name__ == "__main__":
    download_clean()