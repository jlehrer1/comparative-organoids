import pathlib 
import os 
import sys
import argparse
import urllib 

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import helper 

here = pathlib.Path(__file__).parent.absolute()
data_path = os.path.join(here, '..', '..', 'data')

def _download_from_key(key, localpath):
    """
    Helper function that downloads all files recursively from the given key (folder) from the braingeneersdev S3 bucket
    
    Parameters:
    key: S3 folder (key) to start downloading recursively from
    localpath: Optional argument, downloads to a subfolder under the data/processed/ folder # TODO add folder generation
    """

    print(f'Key is {key}')
    reduced_files = helper.list_objects(key)

    for f in reduced_files:
        if not os.path.isfile(os.path.join(data_path, 'processed', f.split('/')[-1])):
            print(f'Downloading {f} from S3')
            helper.download(
                f,
                os.path.join(data_path, 'processed', localpath, f.split('/')[-1]) # Just the file name in the list of objects
            )

def download_clean_expression_matrices() -> None:
    key = os.path.join('jlehrer', 'expression_data', 'processed')
    local_path = os.path.join(data_path, 'processed')

    _download_from_key(key, local_path) 

def download_transposed_expression_matrices() -> None:
    """Downloads the interim data from S3. Interim data is in the correct structural format but has not been cleaned."""
    key = os.path.join('jlehrer', 'expression_data', 'interim')
    local_path = os.path.join(data_path, 'interim')

    _download_from_key(key, local_path)

def download_raw_expression_matrices(upload) -> None:
    """Downloads all raw datasets and label sets, and then unzips them. This will only be used during the data processing step"""

    # {local file name: [dataset url, labelset url]}
    datasets = helper.DATA_FILES_AND_URLS_DICT

    for file, links in datasets.items():
        datafile_path = os.path.join(data_path, 'external', file)

        labelfile = f'{file[:-4]}_labels.tsv'
        labelfile_path = os.path.join(data_path, 'external', labelfile)

        datalink, labellink = links 

        # First, make the required folders if they do not exist 
        for dir in 'external', 'interim', 'processed':
            os.makedirs(os.path.join(data_path, dir), exist_ok=True)

        # Download and unzip data file if it doesn't exist 
        if not os.path.isfile(os.path.join(data_path, 'external', file)):
            print(f'Downloading zipped data for {file}')
            urllib.request.urlretrieve(
                datalink,
                f'{datafile_path}.gz',
            )

            print(f'Unzipping {file}')
            os.system(
                f'zcat < {datafile_path}.gz > {datafile_path}'
            )

            print(f'Deleting compressed data')
            os.system(
                f'rm -rf {datafile_path}.gz'
            )

        # Download label file if it doesn't exist 
        if not os.path.isfile(os.path.join(data_path, 'external', labelfile_path)):
            print(f'Downloading label for {file}')
            urllib.request.urlretrieve(
                labellink,
                labelfile_path,
            )

        # If upload boolean is passed, also upload these files to the braingeneersdev s3 bucket
        if upload:
            print(f'Uploading {file} and {labelfile} to braingeneersdev S3 bucket')
            helper.upload(
                datafile_path,
                os.path.join('jlehrer', 'expression_data', 'raw', file)
            )

            helper.upload(
                labelfile_path,
                os.path.join('jlehrer', 'expression_data', 'raw', f'{file[:-4]}_labels.tsv')
            )

    print('Done.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--type',
        type=str,
        required=False,
        default='clean',
        help="Type of data to download"
    )

    parser.add_argument(
        '--s3-upload',
        required=False,
        action='store_true',
        help='If passed, also upload data to braingeneersdev/jlehrer/expression_data/raw, if the method accepts this option'
    )

    args = parser.parse_args()
    type = args.type
    upload = args.s3_upload

    if type == 'interim':
        download_transposed_expression_matrices()
    elif type == 'raw' or type == 'zipped':
        download_raw_expression_matrices(upload=upload)
    elif type == 'processed' or type == 'clean':
        download_clean_expression_matrices()
    else:
        raise ValueError('Unknown type specified for data downloading.')