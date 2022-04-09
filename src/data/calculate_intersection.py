from email.policy import default
import numpy as np 
import os 
import pathlib 
import sys 
import argparse 
from tqdm import tqdm 

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import helper 
from models.lib.data import GeneExpressionData, clean_sample 

def calculate_intersection(indata, genes, datapath):
    # For each dataset and then for each sample in the dataset, remove the bad indices of the sample and write to file 
     for datafile, labelfile in indata:
        dataset = GeneExpressionData(
            filename=datafile,
            labelname=labelfile,
        )

        currgenes = dataset.get_features()
        outfile = os.path.join(datapath, 'processed', f'{datafile[:-4]}_cleaned.csv')

        with open(outfile, mode='a+', encoding='utf-8') as f:
            for sample, _ in dataset: # Don't need label 
                temp = clean_sample(
                    sample=sample,
                    refgenes=genes,
                    currgenes=currgenes,
                ) 

                f.write('\n'.join(temp))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--file',
        required=False,
        default=None,
        type=str,
        help='Calculate and write out gene intersection of specific file'
    )

    data = helper.INTERIM_DATA_AND_LABEL_FILES_LIST
    datafiles, labelfiles = data.keys(), data.values()
    
    here = pathlib.Path(__file__).parent.resolve()
    data_path = os.path.join(here, '..', 'data')

    genes = helper.gene_intersection()

    for file in datafiles:
        calculate_intersection(
            indata=os.path.join(data_path, 'interim', file),
            outdata=os.path.join(data_path, 'processed', file),
            genes=genes
        )
