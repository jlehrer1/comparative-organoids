from enum import unique
import pathlib 
import os
import sys
import pandas as pd 
import numpy as np 
import argparse
from typing import List 

from sklearn.preprocessing import LabelEncoder
import dask.dataframe as da
from dask.diagnostics import ProgressBar

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import helper 

here = pathlib.Path(__file__).parent.absolute()
data_path = os.path.join(here, '..', '..', 'data')

df1_mapping = {
    'Inhibitory Neuron': 'Interneuron'
}

df2_mapping = {
    'IT': 'Interneuron',
    'L4 IT': 'Interneuron',
    'VIP': 'Interneuron',
    'PVALB': 'Interneuron',
    'L6 CT': 'Interneuron', 
    'LAMP5': 'Interneuron',
    'SST': 'Interneuron',
    'Exclude': 'Exclude',
    'Oligodendrocyte': 'Oligodendrocyte',
    'Astrocyte': 'Astrocyte',
    'L6b': 'Excitatory Neuron',
    'L5/6 IT Car3': 'Excitatory Neuron',
    'L5/6 NP': 'Excitatory Neuron',
    'OPC': 'OPC',
    'Microglia': 'Microglia',
    'PAX6': 'Progenitors', # DROP THESE, progenitors should not be in postnatal 
    'L5 ET': 'Excitatory Neuron',
    'Endothelial': 'Endothelial',
    'Pericyte': 'Pericyte',
    'VLMC': 'Vascular',
}

df3_mapping = {
    'L2/3 IT': 'Excitatory Neuron',
    'L5 IT': 'Interneuron',
    'Pvalb': 'Interneuron',
    'Sst': 'Interneuron',
    'Vip': 'Interneuron',
    'Lamp5': 'Interneuron',
    'L6 CT': 'Interneuron',
    'Oligo': 'Oligodendrocyte',
    'L6b': 'Excitatory Neuron',
    'L6 IT': 'Excitatory Neuron',
    'L5/6 NP': 'Excitatory Neuron',
    'Sncg': 'Interneuron',
    'L5 ET': 'Excitatory Neuron',
    'Astro': 'Astrocyte',
    'L6 IT Car3': 'Excitatory Neuron',
    'OPC': 'OPC',
    'Micro-PVM': 'Microglia',
    'Sst Chodl': 'Interneuron',
    'Endo': 'Endothelial',
    'VLMC': 'Vascular',
}

df4_mapping = {
    'RG' : 'Radial Glia', 
    'CR': 'Cajal Retzius',
}

def clean_labelsets(upload: bool) -> None:
    """
    Combines all label sets to be consistent, and writes this to disc. 

    Parameters:
    upload: Whether or not to upload cleaned data to braingeneersdev/jlehrer/expression_data/labels
    """

    # Read in our four datasets
    # MAKE SURE THESE ARE IN THE SAME ORDER AS helper.DATA_FILES_LIST
    print('Reading in datasets')
    files = helper.DATA_FILES_LIST
    interim_data_path = os.path.join(here, '..', '..', 'data', 'interim')
    df1 = pd.read_csv(os.path.join(interim_data_path, 'primary_bhaduri_labels.tsv'), sep='\t')
    df2 = pd.read_csv(os.path.join(interim_data_path, 'allen_cortex_labels.tsv'), sep='\t')
    df3 = pd.read_csv(os.path.join(interim_data_path, 'allen_m1_region_labels.tsv'), sep='\t')
    df4 = pd.read_csv(os.path.join(interim_data_path, 'whole_brain_bhaduri_labels.tsv'), sep='\t')

    # Create the output directories if they don't exist 
    os.makedirs(os.path.join(data_path, 'processed', 'labels'), exist_ok=True)
    os.makedirs(os.path.join(data_path, 'interim', 'labels'), exist_ok=True)

    # Rename cells so we have label consistency across classes 
    print('Mapping target labels to be consistent')
    df1['Type'] = df1['Type'].replace(df1_mapping)
    df2['subclass_label'] = df2['subclass_label'].replace(df2_mapping)
    df3['subclass_label'] = df3['subclass_label'].replace(df3_mapping)
    df4['celltype'] = df4['celltype'].replace(df4_mapping)

    # Grab only columns of interest and rename them for column name consistency 
    print('Cleaning labels...')

    # Select only by type and rename labels 
    df1_reduced = df1[['Type']]
    df2_reduced = df2[['subclass_label']].rename(columns={'subclass_label': 'Type'})
    df3_reduced = df3[['subclass_label']].rename(columns={'subclass_label': 'Type'})
    df4_reduced = df4[['celltype']].rename(columns={'celltype': 'Type'})

    # Remove all whitespace since we have 'Neuron ', and 'Neuron' in our label sets!
    datasets = [df1_reduced, df2_reduced, df3_reduced, df4_reduced]
    for df in datasets:
        df.loc[:, 'Type'] = df.loc[:, 'Type'].apply(lambda x: x.rstrip())

    # Write out the interim labels before removing samples we don't want to train on
    # Since we want to visualize this
    for idx, filename in enumerate(files):
        df = datasets[idx]
        df.to_csv(os.path.join(data_path, 'interim', 'labels', f'{filename[:-4]}_labels.csv'))

    # Now remove all samples we don't want to train on 
    df1_reduced = df1_reduced[df1_reduced['Type'] != 'Outlier']
    df2_reduced = df2_reduced[(df2_reduced['Type'] != 'Exclude') & (df2_reduced['Type'] != 'Progenitor')]
    df3_reduced = df3_reduced[df3_reduced['Type'] != 'Outlier']
    df4_reduced = df4_reduced[df4_reduced['Type'] != 'Outlier']

    # Fit a labelencoder on the intersection of the targets
    unique_targets = list(set(np.concatenate([df['Type'].unique() for df in datasets])))
    le = LabelEncoder()
    le = le.fit(unique_targets)

    # Make a list of our four datasets to index when we are encoding them
    datasets = [df1_reduced, df2_reduced, df3_reduced, df4_reduced]

    # Categorically encode the targets and 
    # Write out the numerically encoded targets to disk 
    # when we read in, set index_col='cell'
    for idx, filename in enumerate(files):
        df = datasets[idx]
        df['Type'] = le.transform(df['Type'])
        df.index.name = 'cell'
        df.to_csv(os.path.join(data_path, 'processed', 'labels', f'{filename[:-4]}_labels.csv'))

        if upload:
            helper.upload(
                os.path.join(data_path, 'labels', filename),
                os.path.join('jlehrer', 'expression_data', 'labels', filename)
            )

def clean_datasets(
    upload: bool,
    single_file: str=None 
) -> None:
    """
    Cleans the gene expression datasets by taking the intersection of columns (genes) between them, and then sorting the columns to make sure that each dimension of the output Tensor corresponds to the same gene. These are read AFTER the expression matrices have been transposed, and will throw an error if these files don't exist in data/interim/

    Parameters:
    upload: Whether or not to upload cleaned data to braingeneersdev/jlehrer/expression_data/data
    """

    unique = helper.gene_intersection()

    if not single_file:
        files = helper.DATA_FILES_LIST
        files = [f'{file[:-4]}_T.csv' for file in files]
    else:
        files = [f'{single_file[:-4]}_T.csv']

    # Get rid of LOC genes before subsetting 
    print(f'Number of unique genes across all datasets are {len(unique)}')
    print(f'Sorting columns and calculating intersection of Dask dataframes')

    for file in files:
        print(f'Calculating for {file}')
        data = (da.read_csv(
            os.path.join(data_path, 'interim', file),
            assume_missing=True,
            header=1, # need this since calculating transpose adds one extra row, still need to figure out why as it doesn't happen on synthetic data 
            sample=1000000, # May need to make this bigger
        ))
        data.columns = [x.split('|')[0].upper() for x in data.columns]
        
        print(f'Number of unique genes is {len(unique)} and number of genes in current dataset is {len(data.columns)}')
        for gene in data.columns:
            if gene not in unique:
                print(f'Dropping gene {gene}')
                data = data.drop(gene, axis=1)
                data = data.persist()

        data.to_csv(
            os.path.join(data_path, 'processed', f'{file[:-4]}.csv'),
            single_file=True,
            index=False,
        )

        print(f'Uploading {file} to S3')
        if upload:
            helper.upload(
                os.path.join(data_path, 'processed', f'{file[:-4]}.csv'),
                os.path.join('jlehrer', 'expression_data', 'clean', f'{file[:-4]}.csv')
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--labels',
        required=False,
        help='If passed, run the code for label cleaning, otherwise, continue.',
        action='store_true',
    )

    parser.add_argument(
        '--data',
        required=False,
        help='If passed, run the code for data cleaning. This should be run remotely, as it requires the interaction between large Dask dataframes which will likely be quite slow. Otherwise, continue',
        action='store_true',
    )

    parser.add_argument(
        '--s3-upload',
        required=False,
        help='If passed, also upload the results to the braingeneersdev S3 bucket',
        action='store_true'
    )

    parser.add_argument(
        '--generate-genes',
        required=False,
        help='If passed, generate the intersection of genes to data/',
        action='store_true',
    )

    parser.add_argument(
        '--file',
        required=False,
        help='If passed, calculate the gene intersection for the passed file. Otherwise, calculate the gene intersection for all data. This exists so this script can be run in parallel',
        type=str,
    )

    args = parser.parse_args()
    file = args.file 
    labels = args.labels
    data = args.data 
    upload = args.s3_upload
    generate_genes = args.generate_genes

    if file and not data:
        raise ValueError("--data flag must also be passed if calculating gene intersection for specific file.")


    if data and file: 
        print(f'Calculating gene intersection for {file}')
        clean_datasets(upload=upload, single_file=file)
    if data and not file: 
        clean_datasets(upload=upload)
    if labels: 
        clean_labelsets(upload=upload)
    if generate_genes: 
        import json 
        genes = json.dumps(helper.gene_intersection())
        with open(os.path.join(data_path, 'genes.txt'), 'w', encoding='utf-8') as f:
            json.dump(genes, f, ensure_ascii=False, indent=4)