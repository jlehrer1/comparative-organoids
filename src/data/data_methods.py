import pathlib 
import os 
import pandas as pd 
import numpy as np 
import sys 
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from sklearn.preprocessing import LabelEncoder
import dask.dataframe as da
from dask.diagnostics import ProgressBar
from helper import upload

pbar = ProgressBar()
pbar.register() # global registration

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

def combine_labelsets():
    """
    Combines all label sets to be consistent, and writes this to disc. 
    """

    # Read in our four datasets 
    print('Reading in datasets')
    try:
        df1 = pd.read_csv(os.path.join(here, '..', '../data/meta/meta_primary.tsv', sep='\t'))
        df2 = pd.read_csv(os.path.join(here, '..', '../data/external/whole_brain_bhaduri_labels.tsv', sep='\t'))
        df3 = pd.read_csv(os.path.join(here, '..', '../data/external/allen_m1_region_labels.tsv', sep='\t'))
        df4 = pd.read_csv(os.path.join(here, '..', '../data/external/allen_cortex_labels.tsv', sep='\t'))
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

    # Get the union of all targets 
    datasets = [df1_reduced, df2_reduced, df3_reduced, df4_reduced]
    unique_targets = list(set(np.concatenate([df['Type'].unique() for df in datasets])))

    le = LabelEncoder()
    le = le.fit(unique_targets)

    datasets = {
        df1_reduced: 'bhaduri_2020_labels.csv',
        df2_reduced: 'bhaduri_2021_labels.csv',
        df3_reduced: 'allen_m1_region_labels.csv',
        df4_reduced: 'allen_cortex_labels.csv',
    }

    # Categorically encode the targets and 
    # Write out the numerically encoded targets to disk 
    # when we read in, set index_col='cell'

    for df, filename in datasets.items():
        df['Type'] = le.transform(df['Type'])
        df.index.name = 'cell'
        df.to_csv(os.path.join(data_path, 'labels', filename))

def clean_datasets():
    """
    Cleans the gene expression datasets by taking the intersection of columns (genes) between them, and then sorting the columns to make sure that 
    each dimension of the output Tensor corresponds to the same gene. 
    """

    files = [
        'bhaduri_2020_T.csv',
        # 'bhaduri_2021_T.csv',
        'allen_m1_T.csv',
        'allen_cortex_T.csv'
    ]

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
        upload(
            os.path.join(data_path, 'processed', 'data', f'{file[:-4]}.csv'),
            os.path.join('jlehrer', 'expression_data', 'data', f'{file[:-4]}.csv')
        )
