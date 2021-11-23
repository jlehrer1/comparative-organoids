import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import umap
import plotly.express as px
import plotly.graph_objects as go
import hdbscan
import seaborn as sns
import pathlib 
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from helper import upload

here = pathlib.Path(__file__).parent.absolute()
primary = pd.read_csv(os.path.join(here, '../../data/processed/primary_reduction_neighbors_50_components_50.csv', index_col='Unnamed: 0'))
prim_umap = pd.read_csv(os.path.join(here, '../../data/processed/primary_reduction_neighbors_50_components_2.csv', index_col='Unnamed: 0'))

clusterer = hdbscan.HDBSCAN(min_cluster_size=20)
clusters = clusterer.fit(primary)

prim_umap['label'] = clusters.labels_

fig, ax = plt.subplots(figsize=(10, 10))

sns.scatterplot(
    x='0', 
    y='1',
    data=prim_umap,
    hue='label',
    legend='full',
    ax=ax,
    s=1,
    palette='bright'
)

plt.title(f'UMAP Projection of Primary Data, Colored by Cluster (computed on N=50 components)')
plt.savefig('umap_cluster.png', dpi=300)
upload('umap_cluster.png', 'umap_cluster_test.png')
plt.show()