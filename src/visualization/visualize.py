import dask.dataframe as dd
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
import seaborn as sns 
import pathlib 
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

def umap_viz_2d(reduced, save_path, palette='bright') -> None:
    """
    Plots the two dimensional UMAP projection, colored by cluster label

    Parameters:
    data: m x 3 DataFrame, where columns '0' and '1' denote the two UMAP axes, and 'label' is the column with cluster labels.
    save_path: Path to save output figure to
    palette='bright': color pallette for plot
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    sns.scatterplot(
        x='0', 
        y='1',
        data=reduced,
        hue='label',
        legend='full',
        ax=ax,
        s=1,
        palette=palette,
    )
    
    plt.savefig(save_path)

def pairplot(reduced, primary, of_interest, subplot_shape, save_path) -> None:
    """
    Plots the expression levels of the given genes on a 2D umap projection.

    Parameters:
    reduced: DataFrame with the two UMAP axes labeled as '0' and '1'
    primary: Path to the processed primary tissue data
    of_interest: List of genes to plot 
    subplot_shape: Tuple containing (rows, cols) of matplotlib subplots
    save_path: Path to save output figure to
    """

    rows, cols = subplot_shape
    fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(15, 15))
    for i, ax in enumerate(axes.flatten()):
        vals = pd.read_csv(primary, usecols=[of_interest[i].upper()])
        reduced['vals'] = vals
        sns.scatterplot(
            ax=ax,
            x='0',
            y='1',
            data=reduced,
            hue='vals',
            legend=None,
            s=1,
        )
        
        ax.set_title(of_interest[i])

    plt.savefig(save_path)

if __name__ == '__main__':
    here = pathlib.Path(__file__).parent.absolute()
    data_path = os.path.join(here, '..', '..', 'data', 'processed')
    primary_path = os.path.join(data_path, 'primary.csv')

    ann_files = [f.rstrip() for f in os.listdir(os.path.join(data_path, 'annotations')) if f.startswith('annotation_primary_labels_neighbors')] # To make sure we're only considering the UMAP-based clustering, not the PCA 
    clust_files = [f.rstrip() for f in os.listdir(os.path.join(data_path, 'labels')) if f.startswith('primary_labels')]

    print(clust_files)
    reduced = pd.read_csv(os.path.join(data_path, 'umap', 'primary_reduction_neighbors_100_components_2.csv')) # For cluster visualization

    for file in clust_files:
        clusters = pd.read_csv(os.path.join(data_path, 'labels', file)).loc[:, '# label']
        reduced['label'] = clusters 

        name = file.split('.')[0]
        umap_viz_2d(
            reduced=reduced,
            save_path=os.path.join(here, f'{name}_visualization.png')
        )
        
    of_interest = ['id2', 'sox5', 'tbr1', 'sox2', 'dcx', 'fezf2', 'plxnd1', 'gfra2', 'gad1']

    pairplot(
        reduced=reduced,
        primary=primary_path,
        of_interest=of_interest,
        subplot_shape=(3,3),
        save_path=os.path.join(here, 'atlas_gene_plot.png')
    )

