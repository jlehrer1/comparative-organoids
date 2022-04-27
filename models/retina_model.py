import os
import pathlib 
import sys
import anndata as an
import torch 
import argparse 

from os.path import join, dirname, abspath
sys.path.append(join(dirname(abspath(__file__)), '..', 'src'))

from helper import download
from models.lib.lightning_train import generate_trainer
from models.lib.data import total_class_weights

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

trainer, model, module = generate_trainer(
    datafiles=datafiles,
    labelfiles=labelfiles,
    class_label='class_label',
    index_col='cell',
    drop_last=True,
    shuffle=True,
    normalize=True,
    batch_size=4,
    num_workers=0,
    weighted_metrics=True,
    optim_params = {
        'optimizer': torch.optim.Adam,
        'lr': lr,
        'weight_decay': weight_decay,
    },
    scheduler_params={
        'scheduler': torch.optim.lr_scheduler.StepLR,
        'step_size': 30,
        'gamma': 0.001,
    },
    max_epochs=500,
    skip=3,
    weights=total_class_weights([join(data_path, 'retina_labels_numeric.csv')], 'class_label'),
    wandb_name=name,
)

# train model
trainer.fit(model, datamodule=module)