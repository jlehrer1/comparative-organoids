from xgboost import XGBClassifier
import anndata as an
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 

import sys
sys.path.append('../src')
from models.lib.lightning_train import DataModule
import xgboost as xgb
from sklearn.metrics import accuracy_score 

module = DataModule(
    datafiles=['../data/retina/retina_T.h5ad'],
    labelfiles=['../data/retina/retina_labels_numeric.csv'],
    class_label='class_label',
    index_col='cell',
    batch_size=16,
    num_workers=0,
    shuffle=True,
    drop_last=False,
    normalize=False,
)
module.setup()

loader = module.trainloader 

model = XGBClassifier()
train = module.trainloader
test = module.valloader

# Train on one minibatch to get started 
sample = next(iter(loader))
X = sample[0].numpy()
y = sample[1].numpy()

model = model.fit(X, y)

for i, (trainsample, valsample) in enumerate(zip(train, test)):
    X_train, y_train = trainsample 
    X_test, y_test = valsample
    
    model = model.fit(X_train, y_train, xgb_model=model.get_booster())
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)
