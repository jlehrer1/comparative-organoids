import dask.dataframe as dd
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
import seaborn as sns 

def two_plot(data, palette='bright'):
    fig, ax = plt.subplots(figsize=(10, 10))

    sns.scatterplot(
        x='0', 
        y='1',
        data=data,
        hue='label',
        legend=None,
        ax=ax,
        s=1,
        palette=palette,
    )
    plt.show()

def pairplot(reduced, primary, of_interest):
    n = len(of_interest)
    fig, axes = plt.subplots(2, 5, sharex=True, sharey=True, figsize=(15, 10))
    for i, ax in enumerate(axes.flatten()):
        try:
            vals = pd.read_csv('../data/processed/primary.csv', usecols=[of_interest[i].upper()])
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
        except Exception as e:
            print(e)
        
        ax.set_title(of_interest[i])

def inhib_excit(reduced, primary):
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    rows = range(0, 10000)

    vals = pd.read_csv('../data/processed/primary.csv', usecols=['DCX'])
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

    vals = pd.read_csv('../data/processed/primary.csv', usecols=['SOX2'])
    reduced['vals'] = vals

    sns.scatterplot(
        ax=ax2,
        x='0',
        y='1',
        data=reduced,
        hue='vals',
        legend=None,
        s=1,
    )
