import os, sys 
import pathlib 
import argparse

from os.path import join, abspath, dirname
from bigcsv import BigCSV 

sys.path.append(join(dirname(abspath(__file__)), '..'))
from helper import upload

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=str,
        help="Specific file to calculate hdf5 of, otherwise calculate all sequentially. This is to be used for parallelism.",
        required=False,
        default=None,
    )

    parser.add_argument(
        "--sep",
        type=str,
        help="File separator on file passed",
        required=False,
        default="\t",
    )

    parser.add_argument(
        "--chunksize",
        type=int,
        help="Chunksize for splitting file before calculating h5ad versions",
        default=400,
        required=False,
    )

    here = pathlib.Path(__file__).parent.absolute()
    data_path = join(here, '..', '..', 'data', 'raw')

    args = parser.parse_args()

    file = args.file
    sep = args.sep 
    chunksize = args.chunksize 
    
    if file is None:
        to_convert = [f for f in os.listdir(data_path) if f.endswith('.tsv')]

        for file in to_convert:
            outfile = pathlib.Path(file).stem + ".h5ad"
            print(f"Converting {outfile}")

            converter = BigCSV(
                file=join(data_path, file),
                outfile=join(data_path, outfile),
                insep=sep,
                chunksize=chunksize,
                quiet=False,
                chunkfolder=join(here, 'converter_chunks'),
                save_chunks=False,
            )

            converter.to_h5ad(sparsify=True)

            upload(
                file_name=outfile,
                remote_name=join('jlehrer', 'h5ad_matrices', outfile)
            )

    else:
        outfile = pathlib.Path(file).stem + ".h5ad"
        print(pathlib.Path(file).stem)
        print(f"Converting {outfile}")
        converter = BigCSV(
            file=join(data_path, file),
            outfile=outfile,
            insep=sep,
            chunksize=chunksize,
            quiet=False,
            chunkfolder=os.path.join(here, 'converter_chunks'),
            save_chunks=False,
        )

        converter.to_h5ad(sparsify=True)

        upload(
            file_name=outfile,
            remote_name=join('jlehrer', 'h5ad_matrices', outfile)
        )