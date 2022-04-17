import scanpy as sc
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder 
import torch
from sklearn.model_selection import train_test_split

import os, sys
sys.path.append('./src')
from src.models.lib.data import *
labelfile = 'data/mouse/Adult Inhibitory Neurons in Mouse_labels.tsv'

mouse_atlas = sc.read_h5ad('data/mouse/MouseAdultInhibitoryNeurons.h5ad')
label_df = pd.read_csv(labelfile, sep='\t')

mo_data = sc.read_h5ad('data/mouse/Mo_PV_paper_TDTomato_mouseonly.h5ad')

X_train, X_test, y_train, y_test = train_test_split(mouse_atlas.X, label_df['numeric_class'].values)

atlas_train = NumpyStreamble(
    matrix=X_train,
    labels=y_train,
    sep='\t',
    columns=list(mouse_atlas.var.index),
    class_label='ingore'
)