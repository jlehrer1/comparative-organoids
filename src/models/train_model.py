import pandas as pd 
import sklearn as sk
import numpy as np
import dask.dataframe as dd
import matplotlib.pyplot as plt 
import pathlib 
import os 

here = pathlib.Path(__file__).parent.absolute()

# Read in the data assuming it is transposed (rows are cells, columns are genes)
df_primary = dd.read_csv(
    os.path.join(here, '..', '..', 'data', 'processed', 'primary.csv')
)

df_organoid = dd.read_csv(
    os.path.join(here, '..', '..', 'data', 'processed', 'organoid.csv')
)



