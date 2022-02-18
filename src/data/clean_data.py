import pathlib 
import os
import sys
import pandas as pd 
import numpy as np 
import argparse

from sklearn.preprocessing import LabelEncoder
import dask.dataframe as da
from dask.diagnostics import ProgressBar

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import helper 

here = pathlib.Path(__file__).parent.absolute()
data_path = os.path.join(here, '..', '..', 'data')

df2_mapping = {
    'RG' : 'Radial Glia', 
    'CR': 'Cajal Retzius'
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
    'PAX6' : 'Progenitors', # DROP THESE, progenitors should not be in postnatal 
    'L5 ET': 'Excitatory Neuron',
    'Endothelial' : 'Endothelial',
    'Pericyte': 'Pericyte',
    'VLMC': 'Vascular'
}

def clean_labelsets(upload: bool) -> None:
    """
    Combines all label sets to be consistent, and writes this to disc. 

    Parameters:
    upload: Whether or not to upload cleaned data to braingeneersdev/jlehrer/expression_data/labels
    """

    # Read in our four datasets 
    print('Reading in datasets')
    raw_data_path = os.path.join(here, '..', 'data', 'raw')
    try:
        df1 = pd.read_csv(os.path.join(raw_data_path, 'primary_bhaduri_labels.tsv', sep='\t'))
        df2 = pd.read_csv(os.path.join(raw_data_path, 'whole_brain_bhaduri_labels.tsv', sep='\t'))
        df3 = pd.read_csv(os.path.join(raw_data_path, 'allen_m1_region_labels.tsv', sep='\t'))
        df4 = pd.read_csv(os.path.join(raw_data_path, 'allen_cortex_labels.tsv', sep='\t'))
    except Exception as e:
        print('Error: Missing file. Double check label file names and make sure all are downloaded properly.')
        print(e)

    # Rename cells so we have label consistency across classes 
    print('Mapping target labels to be consistent')
    df2['celltype'] = df2['celltype'].replace(df2_mapping)
    df3['subclass_label'] = df3['subclass_label'].replace(df3_mapping)
    df4['subclass_label'] = df4['subclass_label'].replace(df4_mapping)

    # Grab only columns of interest and rename them for column name consistency 
    print('Reducing ')
    df1_reduced = df1[['Type']]
    df1_reduced = df1_reduced[df1_reduced['Type'] != 'Outlier']

    df2_reduced = df2[['celltype']].rename(columns={'celltype': 'Type'})
    df2_reduced = df2_reduced[df2_reduced['Type'] != 'Outlier']

    df3_reduced = df3[['subclass_label']].rename(columns={'subclass_label': 'Type'})
    df3_reduced = df3_reduced[df3_reduced['Type'] != 'Outlier']

    df4_reduced = df4[['subclass_label']].rename(columns={'subclass_label': 'Type'})
    df4_reduced = df4_reduced[(df4_reduced['Type'] != 'Exclude') & (df4_reduced['Type'] != 'Progenitor')]

    # Get the union of all labels 
    datasets = [df1_reduced, df2_reduced, df3_reduced, df4_reduced]
    unique_targets = list(set(np.concatenate([df['Type'].unique() for df in datasets])))

    # Fit a labelencoder on the intersection of the targets
    le = LabelEncoder()
    le = le.fit(unique_targets)

    datasets = [df1_reduced, df2_reduced, df3_reduced, df4_reduced]
    datasets = dict(zip(datasets, helper.DATA_FILES_LIST))

    # Categorically encode the targets and 
    # Write out the numerically encoded targets to disk 
    # when we read in, set index_col='cell'
    for df, filename in datasets.items():
        df['Type'] = le.transform(df['Type'])
        df.index.name = 'cell'
        df.to_csv(os.path.join(data_path, 'labels', filename))

        if upload:
            helper.upload(
                os.path.join(data_path, 'labels', filename),
                os.path.join('jlehrer', 'expression_data', 'labels', filename)
            )

def clean_datasets(upload: bool) -> None:
    """
    Cleans the gene expression datasets by taking the intersection of columns (genes) between them, and then sorting the columns to make sure that each dimension of the output Tensor corresponds to the same gene. These are read AFTER the expression matrices have been transposed, and will throw an error if these files don't exist in data/interim/

    Parameters:
    upload: Whether or not to upload cleaned data to braingeneersdev/jlehrer/expression_data/data
    """

    files = helper.DATA_FILES_LIST
    files = [f'{file[:-4]}_T.csv' for file in files]

    cols = []
    for file in files:
        # Read in columns, split by | (since some are PVALB|PVALB), and make sure all are uppercase
        temp = pd.read_csv(os.path.join(data_path, 'interim', file), nrows=1, header=1).columns 
        temp = [x.split('|')[0].upper() for x in temp]
        cols.append(set(temp))

    unique = list(set.intersection(*cols))
    unique = sorted(unique)

    print(f'Number of unique genes across all datasets are {len(unique)}')
    print(f'Sorting columns and calculating intersection of Dask dataframes')

    for file in files:
        print(f'Calculating for {file}')
        data = (da.read_csv(
            os.path.join(data_path, 'interim', file),
            assume_missing=True,
            header=1, # need this since caculating transpose adds one extra row, still need to figure out why as it doesn't happen on synthetic data 
            sample=1000000, # May need to make this bigger
        ))

        data.columns = [x.split('|')[0].upper() for x in data.columns]
        data = data[unique]
        data = data.persist()

        data.to_csv(
            os.path.join(data_path, 'processed', 'data', f'{file[:-4]}.csv'),
            single_file=True,
            index=False,
        )

        print(f'Uploading {file} to S3')

        if upload:
            helper.upload(
                os.path.join(data_path, 'processed', 'data', f'{file[:-4]}.csv'),
                os.path.join('jlehrer', 'expression_data', 'data', f'{file[:-4]}.csv')
            )
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--labels',
        required=False,
        default=None,
        help='If passed, run the code for label cleaning, otherwise, continue.',
        action='store_true',
    )

    parser.add_argument(
        '--data',
        required=False,
        default=None,
        help='If passed, run the code for data cleaning. This should be run remotely, as it requires the interaction between large Dask dataframes which will likely be quite slow. Otherwise, continue',
        action='store_true',
    )

    parser.add_argument(
        '--s3-upload',
        required=False,
        action='store_true'
    )

    args = parser.parse_args()
    labels = args.labels
    data = args.data 
    upload = args.s3_upload

    if not labels and not data:
        print('Nothing arguments passed. Done.')

    if data: 
        clean_datasets(upload=upload)
    if labels: 
        clean_labelsets(upload=upload)


