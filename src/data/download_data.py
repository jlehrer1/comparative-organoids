from multiprocessing.sharedctypes import Value
import pathlib 
import os 
import sys
import argparse
import urllib 

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import helper 

here = pathlib.Path(__file__).parent.absolute()
data_path = os.path.join(here, '..', '..', 'data')

files = [
    'primary.tsv',
    'allen_cortex.tsv',
    'allen_m1_region.tsv',
    'whole_brain_bhaduri.tsv',
]

def _download_from_key(key, localpath=''):
    """
    Helper function that downloads all files recursively from the given key (folder) from the braingeneersdev S3 bucket
    
    Parameters:
    key: S3 folder (key) to start downloading recursively from
    localpath: Optional argument, downloads to a subfolder under the data/processed/ folder # TODO add folder generation
    """

    if localpath != '':
        print(f'Making path {localpath}')
        pathlib.Path(os.path.join(data_path, 'processed', localpath), exist_ok=True)

    print(f'Key is {key}')
    reduced_files = helper.list_objects(key)

    for f in reduced_files:
        if not os.path.isfile(os.path.join(data_path, 'processed', f.split('/')[-1])):
            print(f'Downloading {f} from S3')
            helper.download(
                f,
                os.path.join(data_path, 'processed', localpath, f.split('/')[-1]) # Just the file name in the list of objects
            )

def download_clean(type=None) -> None:
    """
    Downloads the cleaned organoid and primary cell dataset
    """
    if type == 'organoid':
        print(f'Downloading organoid from S3')
        helper.download(
            os.path.join('jlehrer', 'organoid.csv'), 
            os.path.join(data_path, 'processed', 'organoid.csv')
        )
    elif type == 'primary':
        print(f'Downloading primary from S3')
        helper.download(
            os.path.join('jlehrer', 'primary.csv'), 
            os.path.join(data_path, 'processed', 'primary.csv')
        )

def download_interim() -> None:
    """Downloads the interim data from S3. Interim data is in the correct structural format but has not been cleaned."""

    for f in 'organoid_T.csv', 'primary_T.csv':
        if not os.path.isfile(os.path.join(data_path, 'interim', f)):
            print(f'Downloading {f} data from S3')
            helper.download(
                os.path.join('jlehrer', 'transposed_data', f), 
                os.path.join(data_path, 'interim', f)
            )

def download_raw(upload) -> None:
    """Downloads all raw datasets and label sets, and then unzips them. This will only be used during the data processing step"""

    # {local file name: [dataset url, labelset url]}
    datasets = helper.DATA_FILES_AND_URLS_DICT

    for file, links in datasets.items():
        filename = os.path.join(data_path, 'external', file)
        labelname = os.path.join(data_path, 'external', f'{file[:-4]}_labels.tsv')
        datalink, labellink = links 

        # First, make the required folders if they do not exist 
        for dir in 'external', 'interim', 'processed':
            os.makedirs(os.path.join(data_path, dir), exist_ok=True)

        # Download and unzip data file if it doesn't exist 
        if not os.path.isfile(os.path.join(data_path, 'external', file)):
            print(f'Downloading zipped data for {file}')
            urllib.request.urlretrieve(
                datalink,
                f'{filename}.gz',
            )

            print(f'Unzipping {file}')
            os.system(
                f'zcat < {filename}.gz > {filename}'
            )

        # Download label file if it doesn't exist 
        if not os.path.isfile(os.path.join(data_path, 'external', labelname)):
            print(f'Downloading label for {file}')
            urllib.request.urlretrieve(
                labellink,
                labelname,
            )

        # If upload boolean is passed, also upload these files to the braingeneersdev s3 bucket
        if upload:
            print(f'Uploading {file} and {labelname} to braingeneersdev S3 bucket')
            helper.upload(
                filename,
                os.path.join('jlehrer', 'expression_data', 'raw', file)
            )

            helper.upload(
                labelname,
                os.path.join('jlehrer', 'expression_data', 'raw', f'{file[:-4]}_labels.tsv')
            )

    print('Done')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--type',
        type=str,
        required=False,
        default='clean',
        help="Type of data to download. Can be one of ['clean', 'interim', 'raw', 'reduced', 'labels']"
    )

    parser.add_argument(
        '--s3-upload',
        required=False,
        action='store_true',
        help='If passed, also upload data to braingeneersdev/jlehrer/expression_data/raw'
    )

    args = parser.parse_args()
    type = args.type
    upload = args.s3_upload

    if type == 'clean':
        download_clean()
    elif type == 'organoid':
        download_clean('organoid')
    elif type == 'primary':
        download_clean('primary')
    elif type == 'interim':
        download_interim()
    elif type == 'raw' or type == 'zipped':
        download_raw(upload=upload)
    else:
        raise ValueError('Unknown type specified for data downloading.')