import os
import pathlib 
import os, sys
import pytorch_lightning as pl 

sys.path.append(os.path.join(os.path.abspath(__file__), '..', 'src'))
from src.models.lib.data import *
from src.models.lib.lightning_train import DataModule

here = pathlib.Path(__file__).parent.resolve()

module = DataModule(
    datafiles=[os.path.join(here, '..', 'data', 'mouse', 'MouseAdultInhibitoryNeurons.h5ad')],
    labelfile=[os.path.join(here, '..', 'data', 'mouse', 'Adult Inhibitory Neurons in Mouse_labels.tsv')],
    class_label='numeric_class',
    sep='\t',
)

