import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import umap
import plotly.express as px
import plotly.graph_objects as go
import dask.dataframe as dd
import pathlib 
import os 

here = pathlib.Path(__file__).parent.absolute()

# Test reading in compressed files in my Docker container 
dask_test = dd.read_csv(os.path.join(here, '../data/raw/organoid.tsv.gz'), compression='gzip', sep='\t', sample=1000000000)
pandas_test = pd.read_csv(os.path.join(here, '../data/raw/organoid.tsv.gz'), compression='gzip', sep='\t', nrows=10)