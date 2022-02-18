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
        outfile_name = f'{file[:-4]}_T.csv'
        outfile = os.path.join(data_path, 'interim', outfile_name)
        if not os.path.isfile(outfile):
            trans = Transpose(
                file=os.path.join(data_path, 'external', file), 
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
                remote_name=os.path.join('jlehrer', 'interim_expression_data', outfile_name)
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

    args = parser.parse_args()

    chunksize = args.chunksize  
    upload = args.s3_upload 

    transpose_files(
        files=helper.DATA_FILES_LIST,
        chunksize=chunksize,
        upload=upload,
    )