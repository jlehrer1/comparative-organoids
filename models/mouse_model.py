import os
import pathlib 
import sys
import anndata as an

from os.path import join, dirname, abspath
from pytorch_lightning import Trainer 
from pytorch_lightning.loggers import WandbLogger 

sys.path.append(join(dirname(abspath(__file__)), '..', 'src'))

from helper import download, list_objects
from models.lib.neural import GeneClassifier
from models.lib.lightning_train import DataModule

here = pathlib.Path(__file__).parent.resolve()
print('Downloading training and test files')
print(list_objects('jlehrer/mouse_data'))

print('Making data folder')
os.makedirs(join(here, '..', 'data', 'mouse'), exist_ok=True)
for file in ['MouseAdultInhibitoryNeurons.h5ad', 'Mo_PV_paper_TDTomato_mouseonly.h5ad', 'Adult Inhibitory Neurons in Mouse_labels.tsv']:
    print(f'Downloading {file}')
    
    if not os.path.isfile(join(here, '..', 'data', 'mouse', file)):
        download(
            remote_name=join('jlehrer', 'mouse_data', file),
            file_name=join(here, '..', 'data', 'mouse', file),
        )

mouse_atlas = an.read_h5ad(join(here, '..', 'data', 'mouse', 'MouseAdultInhibitoryNeurons.h5ad'))
mo_data = an.read_h5ad(join(here, '..', 'data', 'mouse', 'Mo_PV_paper_TDTomato_mouseonly.h5ad'))

g1 = mo_data.var.index.values
g2 = mouse_atlas.var.index.values
refgenes = sorted(list(set(g1).intersection(g2)))

module = DataModule(
    datafiles=[join(here, '..', 'data', 'mouse', 'MouseAdultInhibitoryNeurons.h5ad')],
    labelfiles=[join(here, '..', 'data', 'mouse', 'Adult Inhibitory Neurons in Mouse_labels.tsv')],
    class_label='numeric_class',
    sep='\t',
    batch_size=4,
    num_workers=0,
    drop_last=True,
    shuffle=True,
    refgenes=refgenes,
    collocate=False,
)

# Read in mo_data since we will eventually test our model on this 
mo_data = join(here, '..', 'data', 'mouse', 'Mo_PV_paper_TDTomato_mouseonly.h5ad')

model = GeneClassifier(
    input_dim=len(refgenes),
    output_dim=module.num_labels,
)

trainer = Trainer()
trainer.fit(model, datamodule=module)