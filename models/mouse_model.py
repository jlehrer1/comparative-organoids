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

data_path = join(pathlib.Path(__file__).parent.resolve(), '..', 'data', 'mouse_data')

print('Making data folder')
os.makedirs(data_path, exist_ok=True)

for file in ['MouseAdultInhibitoryNeurons.h5ad', 'Mo_PV_paper_TDTomato_mouseonly.h5ad', 'MouseAdultInhibitoryNeurons_labels.csv']:
    print(f'Downloading {file}')

    if not os.path.isfile(join(data_path, file)):
        download(
            remote_name=join('jlehrer', 'mouse_data', file),
            file_name=join(data_path, file),
        )

# Calculate gene intersection
mouse_atlas = an.read_h5ad(join(data_path, 'MouseAdultInhibitoryNeurons.h5ad'))
mo_data = an.read_h5ad(join(data_path, 'Mo_PV_paper_TDTomato_mouseonly.h5ad'))

g1 = mo_data.var.index.values
g2 = mouse_atlas.var.index.values

g1 = [x.strip().upper() for x in g1]
g2 = [x.strip().upper() for x in g2]

refgenes = sorted(list(set(g1).intersection(g2)))
print(f"{len(refgenes) = }")

# Define labelfiles and trainer 
datafiles=[join(data_path, 'MouseAdultInhibitoryNeurons.h5ad')]
labelfiles=[join(data_path, 'MouseAdultInhibitoryNeurons_labels.csv')]

trainer, model, module = generate_trainer(
    datafiles=datafiles,
    labelfiles=labelfiles,
    class_label='numeric_class',
    index_col='cell',
    drop_last=True,
    shuffle=True,
    batch_size=32,
    num_workers=64,
    refgenes=refgenes,
    currgenes=g2,
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
    normalize=True,
    weights=total_class_weights([join(data_path, 'MouseAdultInhibitoryNeurons_labels.csv')], 'numeric_class', 'cuda:0'),
    wandb_name=name,
)

# train model
trainer.fit(model, datamodule=module)