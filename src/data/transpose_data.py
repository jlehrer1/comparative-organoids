import os 
import pathlib 
import sys 
import argparse
from typing import List 

import transposecsv

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import helper

def transpose_files(
    files: List[str],
    chunksize: int,
    upload_chunks: bool,
    upload_file: bool,
    no_cache: bool,
) -> None:
    """
    Calculates the tranpose of the given expression matrices from data/raw

    files: List of files to transpose, must all be in data/raw/
    chunksize: Number of lines to read in from each file, defaults to 400
    upload_chunks: Whether to upload the transposed chunks of the file to the S3 bucket under the upload_chunks path
    upload_file: Whether to upload the transposed file to the S3 bucket under the upload_file path 
    no_cache: Whether to recalculate the transpose, regardless of if it exists or not
    """

    here = pathlib.Path(__file__).parent.absolute()
    data_path = os.path.join(here, '..', '..', 'data')

    with open(os.path.join(here, '..', '..', 'credentials')) as f:
        key, access = [line.rstrip() for line in f.readlines()]

    for file in files:
        # Create both the transposed file name, and it's path
        outfile_name = f'{file[:-4]}_T.csv'
        outfile = os.path.join(data_path, 'interim', outfile_name)
        
        # If the file doesn't already exist, use the Transpose API to calculate
        # The transpose and upload to S3, if the upload parameter is passed
        os.makedirs(os.path.join(data_path, 'interim'), exist_ok=True)
        if not os.path.isfile(outfile) or no_cache:
            transpose = transposecsv.Transpose(
                file=os.path.join(data_path, 'raw', file), 
                outfile=outfile,
                sep='\t',
                chunksize=chunksize,
            )
            transpose.compute()

            if upload_file:
                transpose.upload(
                    bucket="braingeneersdev",
                    endpoint_url="https://s3.nautilus.optiputer.net",
                    aws_secret_key_id=key,
                    aws_secret_access_key=access,
                    remote_file_key=os.path.join('jlehrer', 'expression_data', 'interim', outfile_name)
                )

            if upload_chunks:
                transpose.upload(
                    bucket="braingeneersdev",
                    endpoint_url="https://s3.nautilus.optiputer.net",
                    aws_secret_key_id=key,
                    aws_secret_access_key=access,
                    remote_chunk_path=os.path.join('jlehrer', 'expression_data_chunks', f'chunks_{outfile_name}')
                )
        else:
            print(f"{outfile} exists, continuing...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--chunksize',
        required=False,
        type=int,
        default=400
    )

    parser.add_argument(
        '--file',
        required=False,
        default=None,
        type=str,
        help='File to calculate transpose of, to be used if this script is parallelized. If nothing is passed, just calculate the transpose of the entire data file list.'
    )

    parser.add_argument(
        '--upload-chunks',
        required=False,
        action='store_true',
        help='If passed, also upload transposed data chunks (before being combined) to the braingeneers s3 bucket under jlehrer/interim_expression_data/'
    )

    parser.add_argument(
        '--upload-file',
        required=False,
        action='store_true',
        help='If passed, also upload the transposed data to the braingeneers s3 bucket under jlehrer/interim_expression_data/'
    )

    parser.add_argument(
        '--no-cache',
        required=False,
        action='store_true',
        help='If passed, ignore the already existing transposed file and rerun the script'
    )


    args = parser.parse_args()

    chunksize = args.chunksize  
    file = args.file
    upload_chunks = args.upload_chunks 
    upload_file = args.upload_file 
    no_cache = args.no_cache 

    # If files is a str, i.e. a single file to be used when calling this script in parallel, make it into a list of length one 
    # as expected by transpose_files
    if not file:
        files = helper.DATA_FILES_LIST
    else:
        files = [file] 
         
    transpose_files(
        files=files,
        chunksize=chunksize,
        upload_chunks=upload_chunks,
        upload_file=upload_file,
        no_cache=no_cache,
    )