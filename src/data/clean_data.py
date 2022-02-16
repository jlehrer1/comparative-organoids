import pathlib 
import os 
import sys
import pandas as pd 
import numpy as np 

from sklearn.preprocessing import LabelEncoder
import dask.dataframe as da
from dask.diagnostics import ProgressBar

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from helper import upload, S3_CLEAN_DATA_PATH
from download_data import download_interim

pbar = ProgressBar()
pbar.register() # global registration


here = pathlib.Path(__file__).parent.absolute()

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

    # Categorically encode the targets 
    le = LabelEncoder()
    le = le.fit(unique_targets)

    df1_reduced['Type'] = le.transform(df1_reduced['Type'])
    df2_reduced['Type'] = le.transform(df2_reduced['Type'])
    df3_reduced['Type'] = le.transform(df3_reduced['Type'])
    df4_reduced['Type'] = le.transform(df4_reduced['Type'])

    # Write out the numerically encoded targets to disk 
    data_path = os.path.join(here, '..', '..', 'data')

    df1_reduced.to_csv(os.path.join(data_path, 'labels', 'bhaduri_2020_labels.csv'))
    df2_reduced.to_csv(os.path.join(data_path, 'labels', 'bhaduri_2021_labels.csv'))
    df3_reduced.to_csv(os.path.join(data_path, 'labels', 'allen_m1_region_labels.csv'))
    df4_reduced.to_csv(os.path.join(data_path, 'labels', 'allen_cortex_labels.csv'))

if __name__ == "__main__":
    download_interim()

    print('Reading in interim organoid data with Dask')
    organoid = (da.read_csv(
        os.path.join(here, '..', '..', 'data', 'interim', 'organoid_T.csv'), 
        assume_missing=True)
    )

    print('Reading in interim primary data with Dask')
    primary = (da.read_csv(
        os.path.join(here, '..', '..', 'data', 'interim', 'primary_T.csv'), 
        assume_missing=True)
    )

    # Fix gene expression names in organoid data
    print('Fixing organoid column names')
    organoid_cols = [x.split('|')[0] for x in organoid.columns]
    organoid.columns = organoid_cols

    print('Renaming index')
    organoid.index = organoid.index.rename('cell')
    primary.index = primary.index.rename('cell')

    # Consider only the genes between the two
    print('Calculating gene intersection')
    subgenes = list(set(organoid.columns).intersection(primary.columns))

    print(f'Number of intersecting genes is {len(subgenes)}')
    print(f'Type of organoid and primary is {type(organoid)}, {type(primary)}')

    # Just keep those genes
    organoid = organoid.loc[:, subgenes]
    primary = primary.loc[:, subgenes]

    # Fill NaN's with zeros
    print('Filling NaN\'s with zeros')
    organoid = organoid.fillna(0)
    primary = primary.fillna(0)

    print('Doing all computations')
    organoid = organoid.persist()
    primary = primary.persist()

    # Write out files 
    print('Writing out clean organoid data to csv')
    organoid.to_csv(os.path.join(here, '..', '..', 'data', 'processed', 'organoid.csv'), single_file=True, index=False)

    print('Writing out clean primary data to csv')
    primary.to_csv(os.path.join(here, '..', '..', 'data', 'processed', 'primary.csv'), single_file=True, index=False)

    print('Uploading clean data to S3')
    upload(os.path.join(here, '..', '..', 'data', 'processed', 'primary.csv'), os.path.join(S3_CLEAN_DATA_PATH, 'primary.csv'))
    upload(os.path.join(here, '..', '..', 'data', 'processed', 'organoid.csv'), os.path.join(S3_CLEAN_DATA_PATH, 'organoid.csv'))
