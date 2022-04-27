import os
import pathlib 
import sys
import anndata as an
import torch 
import argparse 
import pytorch_lightning as pl 
from pytorch_lightning.loggers import WandbLogger

from os.path import join, dirname, abspath
sys.path.append(join(dirname(abspath(__file__)), '..', 'src'))

from helper import download
from models.lib.data import *
from models.lib.neural import *
from models.lib.lightning_train import *

parser = argparse.ArgumentParser()
parser.add_argument(
    '--lr',
    type=float,
    default=0.02,
    required=False,
)

parser.add_argument(
    '--weight-decay',
    type=float,
    default=3e-4,
    required=False,
)

parser.add_argument(
    '--name',
    type=str,
    default=None,
    required=False,
)

args = parser.parse_args()
lr, weight_decay, name = args.lr, args.weight_decay, args.name
data_path = join(pathlib.Path(__file__).parent.resolve(), '..', 'data', 'retina')

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

device = ('cuda:0' if torch.cuda.is_available() else None)
module = DataModule(
    datafiles=datafiles,
    labelfiles=labelfiles,
    class_label='class_label',
    index_col='cell',
    batch_size=8,
    num_workers=32,
    skip=3,
    shuffle=True,
    drop_last=True,
    normalize=True,
)

model = TabNetLightning(
    input_dim=module.num_features,
    output_dim=module.num_labels,
    optim_params={
        'optimizer': torch.optim.Adam,
        'lr': lr,
        'weight_decay': weight_decay,
    },
    scheduler_params={
        'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau,
        'factor': 0.001,
    },
    weights=total_class_weights(labelfiles, 'class_label', device),
)

wandb_logger = WandbLogger(
    project=f"Retina Model",
    name=name,
)

trainer = pl.Trainer(
    gpus=(1 if torch.cuda.is_available() else 0),
    auto_lr_find=False,
    logger=wandb_logger,
    max_epochs=100,
    gradient_clip_val=0.5,
)

# train model
trainer.fit(model, datamodule=module)