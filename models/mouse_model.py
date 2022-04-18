import os
import pathlib 
import os, sys
from pytorch_lightning import Trainer 

sys.path.append(os.path.join(os.path.abspath(__file__), '..', 'src'))
from ..src.models.lib.neural import GeneClassifier
from ..src.models.lib.lightning_train import DataModule

here = pathlib.Path(__file__).parent.resolve()

module = DataModule(
    datafiles=[os.path.join(here, '..', 'data', 'mouse', 'MouseAdultInhibitoryNeurons.h5ad')],
    labelfiles=[os.path.join(here, '..', 'data', 'mouse', 'Adult Inhibitory Neurons in Mouse_labels.tsv')],
    class_label='numeric_class',
    sep='\t',
    batch_size=4,
    num_workers=0,
    drop_last=True,
    shuffle=True,
)

model = GeneClassifier(
    input_dim=module.num_features,
    output_dim=module.num_labels,
)

trainer = Trainer()

trainer.fit(model, datamodule=module)

