import pathlib 
import os 
import sys
import argparse
import urllib 
from typing import *

from os.path import join, isfile, isdir 
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import helper 

here = pathlib.Path(__file__).parent.absolute()

def download_raw_expression_matrices(
    datasets: Dict[str, Tuple[str, str]]=None,
    upload: bool=False,
    unzip: bool=True,
    datapath: str=None
) -> None:
    """Downloads all raw datasets and label sets from cells.ucsc.edu, and then unzips them locally

    :param datasets: uses helper.DATA_FILES_AND_URLS_DICT if None. Dictionary of datasets such that each key maps to a tuple containing the expression matrix csv url in the first element,
                    and the label csv url in the second url, defaults to None
    :type datasets: Dict[str, Tuple[str, str]], optional
    :param upload: Whether or not to also upload data to the braingeneersdev S3 bucket , defaults to False
    :type upload: bool, optional
    :param unzip: Whether to also unzip expression matrix, defaults to False
    :type unzip: bool, optional
    :param datapath: Path to folder to download data to. Otherwise, defaults to data/
    :type datapath: str, optional
    """    
    # {local file name: [dataset url, labelset url]}
    datasets = (datasets if datasets is not None else helper.DATA_FILES_AND_URLS_DICT)
    data_path = (datapath if datapath is not None else join(here, '..', '..', '..', 'data', 'raw'))

    if not isdir(data_path):
        print(f'Generating data path {data_path}')
        os.makedirs(data_path, exist_ok=True)

    for file, links in datasets.items():
        labelfile = f'{file[:-4]}_labels.tsv'
        datalink, _ = links

        datafile_path = join(data_path, file)
        # First, make the required folders if they do not exist 
        for dir in 'raw':
            os.makedirs(join(data_path, dir), exist_ok=True)

        # Download and unzip data file if it doesn't exist 
        if not isfile(datafile_path):
            print(f'Downloading zipped data for {file}')
            urllib.request.urlretrieve(
                datalink,
                f'{datafile_path}.gz',
            )

            if unzip:
                print(f'Unzipping {file}')
                os.system(
                    f'zcat < {datafile_path}.gz > {datafile_path}'
                )

                print(f'Deleting compressed data')
                os.system(
                    f'rm -rf {datafile_path}.gz'
                )

        # If upload boolean is passed, also upload these files to the braingeneersdev s3 bucket
        if upload:
            print(f'Uploading {file} and {labelfile} to braingeneersdev S3 bucket')
            helper.upload(
                datafile_path,
                join('jlehrer', 'expression_data', 'raw', file)
            )

def download_labels(
    datasets: Dict[str, Tuple[str, str]]=None,
    upload: bool=False,
    datapath: str=None,
) -> None:
    """Downloads raw label files from given Dictionary

    :param datasets: Dictionary containing the datafile name as the key, and a tuple of the data download url and label download url as the value, defaults to None
    :type datasets: Dict[str, Tuple[str, str]], optional
    :param upload: Whether to upload data to S3, defaults to False
    :type upload: bool, optional
    :param datapath: Path to download data, defaults to None
    :type datapath: str, optional
    """    
    datasets = helper.DATA_FILES_AND_URLS_DICT
    data_path = (datapath if datapath is not None else os.path.join(here, '..', '..', '..', 'data', 'raw', 'labels'))
    
    if not os.path.isdir(data_path):
        os.makedirs(data_path, exist_ok=True)

    for labelfile, (_, labellink) in datasets.items():
        labelfile_path = os.path.join(data_path, f"{labelfile[:-4]}_labels.tsv")

        # Download label file if it doesn't exist 
        if not os.path.isfile(labelfile_path):
            print(f'Downloading label for {labelfile}')
            urllib.request.urlretrieve(
                labellink,
                labelfile_path,
            )
        else:
            print(f'{labelfile} exists, continuing...')

        if upload:
            helper.upload(
                labelfile_path,
                os.path.join('jlehrer', 'expression_data', 'raw', f'{labelfile[:-4]}_labels.tsv')
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data',
        required=False,
        help="If passed, download raw expression matrices",
        action='store_true'
    )

    parser.add_argument(
        '--labels',
        required=False,
        help="If passed, download raw labelfiles",
        action='store_true'
    )

    parser.add_argument(
        '--no-unzip',
        required=False,
        help="If passed, expression matrices won't be unzipped",
        action='store_false'
    )

    parser.add_argument(
        '--s3-upload',
        required=False,
        action='store_true',
        help='If passed, also upload data to braingeneersdev/jlehrer/expression_data/raw, if the method accepts this option'
    )

    args = parser.parse_args()

    data, labels = args.data, args.labels 
    upload = args.data 
    no_unzip = args.no_unzip
    upload = args.s3_upload 

    # if not data and no_unzip:
    #     raise ValueError("--no-unzip option invalid for label set download, since label sets are not compressed")
        
    if data:
        download_raw_expression_matrices(
            upload=upload, 
            unzip=no_unzip
        )
    if labels:
        download_labels(
            upload=upload,
        )
        

