import pandas as pd
import pathlib 
import os 

here = pathlib.Path(__file__).parent.absolute()

NCOLS = 16774 # The exact number of columns


batch_size = 50
from_file = os.path.join(here, '..', '..', 'data', 'raw', 'organoid.tsv')
to_file = os.path.join(here, 'organoid_transposed_in_memory.csv')

for batch in range(NCOLS//batch_size + bool(NCOLS%batch_size)):
    lcol = batch * batch_size
    rcol = min(NCOLS, lcol+batch_size)
    data = pd.read_csv(from_file, usecols=range(lcol, rcol))
    with open(to_file, 'a') as _f:
        data.T.to_csv(_f, header=False)