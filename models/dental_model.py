import os
import pathlib 
import sys
import anndata as an
import torch 
import argparse 

from os.path import join, dirname, abspath
sys.path.append(join(dirname(abspath(__file__)), '..', 'src'))

from helper import download
from data.downloaders.external_download import download_raw_expression_matrices
from models.lib.data import *
from models.lib.neural import *
from models.lib.lightning_train import *
import pytorch_lightning as pl 
from pytorch_lightning.loggers import WandbLogger

urls = {
    'human_dental.tsv': [
        'https://cells.ucsc.edu/dental-cells/human-adult-molars/exprMatrix.tsv.gz',
        'https://cells.ucsc.edu/dental-cells/human-adult-molars/meta.tsv'
    ]
}

download_raw_expression_matrices(urls, unzip=True)