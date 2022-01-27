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
le = le.fit(meta_primary['Subtype'])

meta_trainable['# label'] = le.transform(meta_primary['Subtype'])
meta_trainable['Subtype'] = le.inverse_transform(meta_trainable['# label'])

meta_trainable.to_csv(os.path.join(data_path, 'processed', 'labels', 'meta_primary_labels.csv'), sep=',')