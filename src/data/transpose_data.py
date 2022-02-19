from stat import filemode
from transposecsv import Transpose 
import os 
import pathlib 
import sys 
import argparse

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import helper 

def transpose_files(files, chunksize, upload):
    here = pathlib.Path(__file__).parent.absolute()
    data_path = os.path.join(here, '..', '..', 'data')

    for file in files:
        # Create both the transposed file name, and it's path
        outfile_name = f'{file[:-4]}_T.csv'
        outfile = os.path.join(data_path, 'interim', outfile_name)
        
        # If the file doesn't already exist, use the Transpose API to calculate
        # The transpose and upload to S3, if the upload parameter is passed 
        if not os.path.isfile(outfile):
            trans = Transpose(
                file=os.path.join(data_path, 'raw', file), 
                outfile=outfile,
                sep='\t',
                chunksize=chunksize,
            )
            trans.compute()
        else:
            print(f"{outfile} exists, continuing...")
        if upload:
            print(f'Uploading transposed {file}')
            helper.upload(
                file_name=outfile,
                remote_name=os.path.join('jlehrer', 'expression_data', 'interim', outfile_name)
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--chunksize',
        required=False,
        type=int,
        default=400
    )

    parser.add_argument(
        '--s3-upload',
        required=False,
        action='store_true',
        help='If passed, also upload transposed data to the braingeneers s3 bucket under jlehrer/interim_expression_data/'
    )

    parser.add_argument(
        '--file',
        required=False,
        default=None,
        type=str,
        help='File to calculate transpose of, to be used if this script is parallelized. If nothing is passed, just calculate the transpose of the entire data file list.'
    )

    args = parser.parse_args()

    chunksize = args.chunksize  
    upload = args.s3_upload 
    file = args.file

    # If files is a str, i.e. a single file to be used when calling this script in parallel, make it into a list of length one 
    # as expected by transpose_files
    if not file:
        files = helper.DATA_FILES_LIST
    else:
        files = [file] 
         
    transpose_files(
        files=files,
        chunksize=chunksize,
        upload=upload,
    )