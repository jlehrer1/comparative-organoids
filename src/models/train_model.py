import pandas as pd 
import sklearn as sk
import numpy as np
import dask.dataframe as dd
import matplotlib.pyplot as plt 
import pathlib 
import os 

from dask_ml.xgboost import XGBClassifier

here = pathlib.Path(__file__).parent.absolute()
data_path = os.path.join(here, '..', '..', 'data')

# Check to make sure processed data exists, if not generate_transpose job needs to be run
if not (os.path.isfile(os.path.join(data_path, 'processed', 'primary.csv')) and os.path.isfile(os.path.join(data_path, 'processed', 'organoid.csv'))):
    raise ValueError('Error: processed data does not exist. Please run generate_transpose job and then rerun this script.')

# Read in the data assuming it is transposed (rows are cells, columns are genes)
df_primary = dd.read_csv(
    os.path.join(here, '..', '..', 'data', 'processed', 'primary.csv')
)

df_organoid = dd.read_csv(
    os.path.join(here, '..', '..', 'data', 'processed', 'organoid.csv')
)
