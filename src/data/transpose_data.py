from stat import filemode
from transposecsv import Transpose 
import os 
import pathlib 

if __name__ == "__main__":
    here = pathlib.Path(__file__).parent.absolute()
    data_path = os.path.join(here, '..', '..', 'data')
    external_path = os.path.join(data_path, 'external')

    files = [
        os.path.join(external_path, 'allen_cortex.tsv'),
        os.path.join(external_path, 'allen_m1_region.tsv'),
        os.path.join(external_path, 'whole_brain_bhaduri.tsv'),
    ]
    
    for file in files:
        trans = Transpose(
            file=file, 
            outfile=os.path.join(data_path, 'interim', file),
            sep='\t',
        )

        trans.compute()
    