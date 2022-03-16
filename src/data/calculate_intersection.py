import numpy as np 
import os 
import pathlib 
import sys 
from tqdm import tqdm 

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import helper 

def calculate_intersection(indata, outdata, genes):
    pass 


if __name__ == "__main__":
    datafiles = helper.DATA_FILES_AND_NAMES_DICT.keys()
    here = pathlib.Path(__file__).parent.resolve()
    data_path = os.path.join(here, '..', 'data')

    genes = helper.gene_intersection()

    for file in datafiles:
        calculate_intersection(
            indata=os.path.join(data_path, 'interim', file),
            outdata=os.path.join(data_path, 'processed', file),
            genes=genes
        )
