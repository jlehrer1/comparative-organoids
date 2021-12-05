import dask.dataframe as dd
import pandas as pd 
import torch
import linecache 
import csv
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from dask_ml.model_selection import train_test_split, HyperbandSearchCV
import argparse
import pathlib, os 
torch.manual_seed(0)

class GeneExpressionData(Dataset):
    def __init__(self, filename, labelname):
        self._filename = filename
        self._labelname = labelname
        self._total_data = 0
        
        with open(filename, "r") as f:
            self._total_data = len(f.readlines()) - 1
    
    def __getitem__(self, idx):        
        line = linecache.getline(self._filename, idx + 2)
        label = linecache.getline(self._labelname, idx + 2)
        
        csv_data = csv.reader([line])
        csv_label = csv.reader([label])
        
        data = [x for x in csv_data][0]
        label = [x for x in csv_label][0]
        return torch.from_numpy(np.array([float(x) for x in data])).float(), [int(float(x)) for x in label][0]
    
    def __len__(self):
        return self._total_data
    
    def num_labels(self):
        return pd.read_csv(self._labelname)['# label'].nunique()
    
    def num_features(self):
        return len(self.__getitem__(0)[0])

def fix_labels(file, path):
    labels = pd.read_csv(file)
    labels['# label'] = labels['# label'].astype(int) + 1
    labels.to_csv(os.path.join(path, 'fixed_' + file.split('/')[-1]), index=False)

class NN(nn.Module):
    def __init__(self, N_features, N_labels):
        super(NN, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(N_features, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, N_labels),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device = {device}')

    here = pathlib.Path(__file__).parent.absolute()
    data_path = os.path.join(here, '..', '..', 'data', 'processed')

    fix_labels(os.path.join(data_path, 'primary_labels_neighbors_50_components_50_clust_size_100.csv'), here)
    fixed_labels = pd.read_csv(os.path.join(here, 'fixed_primary_labels_neighbors_50_components_50_clust_size_100.csv'))

    t = GeneExpressionData(
        filename=os.path.join(data_path, 'primary.csv'),
        labelname=os.path.join(here, 'fixed_primary_labels_neighbors_50_components_50_clust_size_100.csv')
    )

    train_size = int(0.8 * len(t))
    test_size = len(t) - train_size

    train, test = torch.utils.data.random_split(t, [train_size, test_size])

    traindata = DataLoader(train, batch_size = 8, num_workers = 6)
    valdata = DataLoader(test, batch_size = 8, num_workers = 6)

    model = NN(
        N_features=t.num_features(),
        N_labels=t.num_labels()
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.08)
    loss_arr = []
    epochs = 1000000

    for i in range(epochs):
        model.train()

        for X, y in traindata:
            X, y = X.to(device), y.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            loss_arr.append(loss.item())
            
        print(f'Loss at epoch {i} is {loss_arr[i]}')