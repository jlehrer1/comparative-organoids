#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import os
import sys
import pandas as pd
import numpy as np
from torch.utils.data import *
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset

sys.path.append('./src/')
sys.path.append('.')

from src.models.lib.neural import GeneClassifier


# In[2]:


from src.models.lib.data import *
from src.helper import *


# In[3]:


# files = os.listdir('../data/processed/labels/')
# files


# In[4]:


# test2 = pd.read_csv('../data/processed/labels/allen_cortex_labels.csv').set_index('cell')
# test2


# In[5]:


# test2.iloc[47508, :]


# In[6]:


test = GeneExpressionData(
    filename='data/interim/allen_cortex_T.csv',
    labelname='data/processed/labels/allen_cortex_labels.csv',
    class_label='Type',
    cast=True,
    skip=3,
)

test.__getitem__(0)


# In[7]:


len(test.labels)



refgenes = gene_intersection()
# In[9]:


len(set(refgenes).intersection(test.columns))


# In[10]:



loader = DataLoader(test, batch_size=4)
sample = next(iter(loader))
sample = sample[0].numpy()


# In[11]:


def clean_sample(sample, refgenes, currgenes):
    intersection = np.intersect1d(currgenes, refgenes, return_indices=True)
    indices = intersection[1] # List of indices in currgenes that equal refgenes 
    
    axis = (1 if sample.ndim == 2 else 0)
    sample = np.sort(sample, axis=axis)
    sample = np.take(sample, indices, axis=axis)

    return torch.from_numpy(sample)


# In[12]:


datafiles, labelfiles = list(INTERIM_DATA_AND_LABEL_FILES_LIST.keys()), list(INTERIM_DATA_AND_LABEL_FILES_LIST.values())

datafiles = [os.path.join('data', 'interim', f) for f in datafiles]
labelfiles = [os.path.join('data', 'processed/labels', f) for f in labelfiles]
datafiles, labelfiles


# In[13]:


train = GeneExpressionData(datafiles[0], labelfiles[0], 'Type', skip=3)
loader = DataLoader(train, batch_size=4)
currgenes = train.columns


# In[14]:


onedsample = train[0][0]
len(onedsample)


# In[15]:


t = (clean_sample(onedsample, refgenes, currgenes))
t


# In[16]:


len(t)


# In[17]:


# twodsample = next(iter(loader))[0]
# twodsample


# In[18]:


# %%timeit

# clean_sample(twodsample, refgenes, currgenes)


# In[ ]:





# In[19]:


# for X, y in tqdm(loader):
#     X = clean_sample(X, refgenes, currgenes)


# In[20]:


sample.ndim


# In[21]:


temp = pd.read_csv(datafiles[0], nrows=1, header=1).columns 


# In[22]:


# cols = []
# for file in datafiles:
#     # Read in columns, split by | (since some are PVALB|PVALB), and make sure all are uppercase
#     temp = pd.read_csv(file, nrows=1, header=1).columns 
#     temp = [x.split('|')[0].upper().strip() for x in temp]
    
#     print(f'Temp is {temp[0:5]}...')
#     cols.append(set(temp))

# unique = list(set.intersection(*cols))
# unique = sorted(unique)


# In[23]:


# len(unique)


# In[24]:


# temp = pd.read_csv(datafiles[0], nrows=1, header=1).columns 
# temp = [x.strip().upper() for x in temp]
# l = train.features


# In[25]:


# l == temp


# In[26]:


# len(set(unique).intersection(l))


# In[27]:


# len(set(unique))


# In[28]:


# len(set(unique).intersection([x.upper().strip() for x in l]))


# In[ ]:





# In[29]:


train = GeneExpressionData(datafiles[0], labelfiles[0], 'Type', skip=3)
loader = DataLoader(train, batch_size=4)

model = GeneClassifier(
    N_features=len(train.columns),
    N_labels=len(train.labels)
)


# In[30]:


sample = next(iter(loader))[0]
sample


# In[31]:


# %%timeit

# model(sample)


# Now let's time iterating over our dataloader with and without the extra data cleaning

# In[32]:


# for X, y in tqdm(loader):
#     X
#     model(X)


# In[33]:


train = GeneExpressionData(datafiles[0], labelfiles[0], 'Type', skip=3)
loader = DataLoader(train, batch_size=4)

model = GeneClassifier(
    N_features=len(refgenes),
    N_labels=len(train.labels)
)

# for X, y in tqdm(loader):
#     X = clean_sample(X, refgenes, train.columns)
#     model(X)


# In[34]:


loader.dataset


# In[35]:


train, test, insize, numlabels, weights = generate_datasets(datafiles, labelfiles, 'Type')


# In[36]:


train.datasets[0].name


# In[37]:


datafiles


# In[38]:


[train.datasets[i].name for i in range(len(datafiles))]


# We can see that it's much faster to clean the sample on each minibatch, since numpy clearly scales well under-the-hood. Therefore, we'll have to write a manual training loop as we can no longer use pytorch lightning.

# In[39]:


from pytorch_lightning import Trainer

# train = GeneExpressionData(datafiles[0], labelfiles[0], 'Type', skip=3)
# loader = DataLoader(train, batch_size=4)

# model = GeneClassifier(
#     N_features=len(train.columns),
#     load
# )


# In[40]:


combined, test, insize, numlabels, weights = generate_datasets(datafiles, labelfiles, 'Type')
numlabels


# In[41]:


train = GeneExpressionData(datafiles[0], labelfiles[0], 'Type', skip=3)
trainloader = DataLoader(train, batch_size=4)

net = GeneClassifier(
    N_features=len(train.columns),
    N_labels=max(train.labels)
)


# In[42]:


loaders = []
refgenes = gene_intersection()


# In[43]:


for datafile, labelfile in zip(datafiles, labelfiles):
    data = GeneExpressionData(
            datafile,
            labelfile,
            'Type',
            cast=False,
    )
    
#     print(data[0][0][0:5])
    loaders.append(data)


# In[44]:


# for data in loaders:
#     print(data.name)
#     print(data[0][0][0:5])


# In[45]:


loaders = [DataLoader(data, batch_size=4) for data in loaders]

# In[46]:


df1 = pd.read_csv('data/interim/allen_cortex_T.csv', nrows=5)
df2 = pd.read_csv('data/interim/allen_m1_region_T.csv', nrows=5)
df3 = pd.read_csv('data/interim/primary_bhaduri_T.csv', nrows=5)
df4 = pd.read_csv('data/interim/whole_brain_bhaduri_T.csv', nrows=5)


# In[47]:


df1


# In[48]:


df2


# In[49]:


df3


# In[50]:


df4


# In[51]:


df1_data = GeneExpressionData(datafiles[0], labelfiles[0], 'Type', cast=False)
df2_data = GeneExpressionData(datafiles[1], labelfiles[1], 'Type', cast=False)
df3_data = GeneExpressionData(datafiles[2], labelfiles[2], 'Type', cast=False)
df4_data = GeneExpressionData(datafiles[3], labelfiles[3], 'Type', cast=False)


# In[ ]:


df4_data.columns


# In[ ]:





# In[ ]:


import torch.optim as optim
import torch.nn as nn

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(100):  # loop over the dataset multiple times
    
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')


# In[ ]:





# In[ ]:





# In[ ]:




