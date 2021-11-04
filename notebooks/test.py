import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import umap
import plotly.express as px
import plotly.graph_objects as go

df_organoid_raw = pd.read_csv('../data/raw/organoid.tsv.gz', compression='gzip', sep='\t', nrows=10)
df_primary_raw = pd.read_csv('../data/raw/primary.tsv.gz', compression='gzip', sep='\t', nrows=10)
df_organoid_raw = df_organoid_raw['gene'].apply(lambda x: x.split('|')[0])