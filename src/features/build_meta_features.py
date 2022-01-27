from tkinter import Label
import pandas as pd 
import os, sys
import pathlib 
from sklearn.preprocessing import LabelEncoder
here = pathlib.Path(__file__).parent.absolute()
data_path = os.path.join(here, '..', '..', 'data')
meta_primary = pd.read_csv(os.path.join(data_path, 'meta', 'meta_primary.tsv'), sep='\t')
meta_trainable = pd.DataFrame()

le = LabelEncoder()

for col in 'Class', 'State', 'Type', 'Subtype':
    meta_trainable[col] = le.fit_transform(meta_primary.loc[:, col])

meta_trainable.to_csv(os.path.join(data_path, 'processed', 'meta_primary_labels.csv'), sep=',')