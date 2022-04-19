import os
import pathlib 
import sys
import anndata as an
import torch 

from os.path import join, dirname, abspath
from pytorch_lightning import Trainer 
from pytorch_lightning.loggers import WandbLogger 

sys.path.append(join(dirname(abspath(__file__)), '..', 'src'))

from helper import download, list_objects
from models.lib.neural import GeneClassifier
from models.lib.lightning_train import DataModule, generate_trainer

here = pathlib.Path(__file__).parent.resolve()
print('Downloading training and test files')
print(list_objects('jlehrer/mouse_data'))

print('Making data folder')
os.makedirs(join(here, '..', 'data', 'mouse_data'), exist_ok=True)
for file in ['MouseAdultInhibitoryNeurons.h5ad', 'Mo_PV_paper_TDTomato_mouseonly.h5ad', 'MouseAdultInhibitoryNeurons_labels.csv']:
    print(f'Downloading {file}')

    if not os.path.isfile(join(here, '..', 'data', 'mouse_data', file)):
        download(
            remote_name=join('jlehrer', 'mouse_data', file),
            file_name=join(here, '..', 'data', 'mouse_data', file),
        )

# Calculate gene intersection
mouse_atlas = an.read_h5ad(join(here, '..', 'data', 'mouse_data', 'MouseAdultInhibitoryNeurons.h5ad'))
mo_data = an.read_h5ad(join(here, '..', 'data', 'mouse_data', 'Mo_PV_paper_TDTomato_mouseonly.h5ad'))

g1 = mo_data.var.index.values
g2 = mouse_atlas.var.index.values

g1 = [x.strip().upper() for x in g1]
g2 = [x.strip().upper() for x in g2]

refgenes = sorted(list(set(g1).intersection(g2)))
print(f"{len(refgenes) = }")

# Define labelfiles and trainer 
datafiles=[join(here, '..', 'data', 'mouse_data', 'MouseAdultInhibitoryNeurons.h5ad')]
labelfiles=[join(here, '..', 'data', 'mouse_data', 'Adult Inhibitory Neurons in Mouse_labels.tsv')]

trainer, model, module = generate_trainer(
    datafiles=datafiles,
    labelfiles=labelfiles,
    class_label='numeric_class',
    drop_last=True,
    shuffle=True,
    batch_size=32,
    num_workers=32,
    refgenes=refgenes,
    currgenes=g2,
    weighted_metrics=True,
    optim_params={
        'optimizer': torch.optim.SGD,
        'lr': 3e-4,
    },
    sep='\t',
    max_epochs=500,
)

# train model
trainer.fit(model, datamodule=module)