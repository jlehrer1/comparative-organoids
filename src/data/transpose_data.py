from stat import filemode
from transposecsv import Transpose 
import os 
import pathlib 

if __name__ == "__main__":
    here = pathlib.Path(__file__).parent.absolute()
    data_path = os.path.join(here, '..', '..', 'data')
    external_path = os.path.join(data_path, 'external')

    files = [
        'primary.tsv',
        'allen_cortex.tsv',
        'allen_m1_region.tsv',
        'whole_brain_bhaduri.tsv',
    ]

    for file in files:
        outfile = os.path.join(data_path, 'interim', f'{file[:-4]}_T.csv')
        if not os.path.isfile(outfile):
            trans = Transpose(
                file=os.path.join(data_path, 'external', file), 
                outfile=outfile,
                sep='\t',
                chunksize=400,
            )
            trans.compute()

        else:
            print(f"{outfile} exists, continuing...")