import os
import pathlib 
import sys
import anndata as an
import torch 

from os.path import join, dirname, abspath
from pytorch_lightning import Trainer 
from pytorch_lightning.loggers import WandbLogger 

sys.path.append(join(dirname(abspath(__file__)), '..', 'src'))

from helper import download
from models.lib.lightning_train import generate_trainer

data_path = join(pathlib.Path(__file__).parent.resolve(), '..', 'data', 'retina_data')

print('Making data folder')
os.makedirs(data_path, exist_ok=True)

for file in ['retina_T.csv', 'retina_labels_numeric.csv']:
    print(f'Downloading {file}')

    if not os.path.isfile(join(data_path, file)):
        download(
            remote_name=join('jlehrer', 'retina_data', file),
            file_name=join(data_path, file),
        )

# Define labelfiles and trainer 
datafiles=[join(data_path, 'retina_T.csv')]
labelfiles=[join(data_path, 'retina_labels_numeric.csv')]

trainer, model, module = generate_trainer(
    datafiles=datafiles,
    labelfiles=labelfiles,
    class_label='class_label',
    drop_last=True,
    shuffle=True,
    batch_size=4,
    num_workers=0,
    weighted_metrics=True,
    optim_params={
        'optimizer': torch.optim.SGD,
        'lr': 0.001,
        'weight_decay': 1e-4,
        'momentum': 1e-4,
    },
    max_epochs=500,
    normalize=True,
    skip=3,
    index_col='cell',
)

# train model
trainer.fit(model, datamodule=module)