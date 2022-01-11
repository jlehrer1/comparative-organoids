import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import os
import dask.dataframe as dd
import pathlib 
import dask.dataframe as dd

here = pathlib.Path(__file__).parent.absolute()
path = os.path.join(here, '..', '..', 'processed')
files = [f.rstrip for f in os.listdir(os.path.join(path, 'labels'))]