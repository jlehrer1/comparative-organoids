import linecache 
import csv
from typing import *
from functools import cached_property

import pandas as pd 
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import sys, os 
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from helper import seed_everything, gene_intersection

seed_everything(42)

class GeneExpressionData(Dataset):
    """
    Defines a PyTorch Dataset for a CSV too large to fit in memory. 

    Init params:
    filename: Path to csv data file, where rows are samples and columns are features
    labelname: Path to label file, where column '# labels' defines classification labels
    class_label: Label to train on, must be in labelname file 
    indices=None: List of indices to use in dataset. If None, all indices given in labelname are used.
    """
    def __init__(
        self, 
        filename: str, 
        labelname: str, 
        class_label: str,
        indices: Iterable[int]=None,
        skip=3,
        cast=True,
        index_col='cell',
        normalize=False,
        map_genes=False,
    ):
        self.filename = filename
        self.labelname = labelname # alias 
        self._class_label = class_label
        self._index_col = index_col 
        self.skip = skip
        self.cast = cast
        self.normalize = normalize 
        self.indices = indices
        self.map_genes = map_genes

        self.refgenes = (gene_intersection() if map_genes else None)

        if indices is None:
            self._labeldf = pd.read_csv(labelname).reset_index(drop=True)
        else:
            self._labeldf = pd.read_csv(labelname).loc[indices, :].reset_index(drop=True)

    def __getitem__(self, idx):
        # Handle slicing 
        if isinstance(idx, slice):
            if idx.start is None or idx.stop is None:
                raise ValueError(f"Error: Unlike other iterables, {self.__class__.__name__} does not support unbounded slicing since samples are being read as needed from disk, which may result in memory errors.")

            step = (1 if idx.step is None else idx.step)
            idxs = range(idx.start, idx.stop, step)
            return [self[i] for i in idxs]

        # The actual line in the datafile to get, corresponding to the number in the self._index_col values 
        data_index = self._labeldf.loc[idx, self._index_col]

        # get gene expression for current cell from csv file
        # We skip some lines because we're reading directly from 
        line = linecache.getline(self.filename, data_index + self.skip)
        
        if self.cast:
            data = torch.from_numpy(np.array(line.split(','), dtype=np.float32)).float()
        else:
            data = np.array(line.split(','))

        if self.normalize:
            data = data / data.max()

        if self.map_genes:
            data = clean_sample(data, self.refgenes, self.columns)

        label = self._labeldf.loc[idx, self._class_label]

        return data, label

    def __len__(self):
        return len(self._labeldf) # number of total samples 

    @property
    def columns(self): # Just an alias...
        return self.features

    @cached_property # Worth caching, since this is a list comprehension on up to 50k strings. Annoying. 
    def features(self):
        data = linecache.getline(self.filename, self.skip - 1)
        data = [x.split('|')[0].upper().strip() for x in data.split(',')]

        return data

    @cached_property
    def labels(self):
        return self._labeldf.loc[:, self._class_label].unique()

    @property
    def shape(self):
        return (self.__len__(), len(self.features))

    @cached_property 
    def class_weights(self):
        labels = self._labeldf.loc[:, self._class_label].values

        return compute_class_weight(
            class_weight='balanced',
            classes=np.unique(labels),
            y=labels
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(filename={self.filename}, labelname={self.labelname})"

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(filename={self.filename}, "
            f"labelname={self.labelname}, "
            f"skip={self.skip}, "
            f"cast={self.cast}, "
            f"normalize={self.normalize}, "
            f"indices={self.indices})"
        )

class SampleLoader(torch.utils.data.DataLoader):
    def __init__(self, refgenes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.refgenes = refgenes
        self.currgenes = self.dataset.columns 
            
    def __iter__(self):
        for batch in super().__iter__():
            yield clean_sample(batch[0], self.refgenes, self.currgenes), batch[1]

def clean_sample(
    sample, 
    refgenes,
    currgenes
) -> torch.Tensor:
    # currgenes and refgenes are already sorted
    # Passed from calculate_intersection

    """
    Remove uneeded gene columns for given sample.

    Arguments:
    sample: np.ndarray
        n samples to clean
    refgenes:
        list of reference genes from helper.generate_intersection(), contains the genes we want to keep
    currgenes:
        list of reference genes for the current sample
    """

    intersection = np.intersect1d(currgenes, refgenes, return_indices=True)
    indices = intersection[1] # List of indices in currgenes that equal refgenes 
    
    axis = (1 if sample.ndim == 2 else 0)
    sample = np.sort(sample, axis=axis)
    sample = np.take(sample, indices, axis=axis)

    return torch.from_numpy(sample)

def generate_datasets(
    datafiles: List[str],
    labelfiles: List[str],
    class_label: str,
    test_prop: float=0.2,
    *args,
    **kwargs,
) -> Tuple[Dataset, Dataset]:
    """
    Generates the COMBINED train/val/test datasets with stratified label splitting. 
    This means that the proportion of each label is the same in the training, validation and test set. 
    
    Parameters:
    datafiles: List of absolute paths to csv files under data_path/ that define cell x expression matrices
    labelfiles: List of absolute paths to csv files under data_path/ that define cell x class matrices
    class_label: Column in label files to train on. Must exist in all datasets, this should throw a natural error if it does not. 
    test_prop: Proportion of data to use as test set 

    Returns:
    Tuple[Dataset, Dataset, Dataset]: Training, validation and test datasets, respectively
    """
    
    train_datasets, val_datasets, test_datasets = [], [], []

    for datafile, labelfile in zip(datafiles, labelfiles):
        # Read in current labelfile
        current_labels = pd.read_csv(labelfile).loc[:, class_label]
        
        # Make stratified split on labels
        trainsplit, valsplit = train_test_split(current_labels, stratify=current_labels, test_size=test_prop)
        trainsplit, testsplit = train_test_split(trainsplit, stratify=trainsplit, test_size=test_prop)

        for indices, data_list in zip([trainsplit, valsplit, testsplit], [train_datasets, val_datasets, test_datasets]):
            data_list.append(
                GeneExpressionData(
                    filename=datafile,
                    labelname=labelfile,
                    class_label=class_label,
                    indices=indices.index,
                    *args,
                    **kwargs,
                )
            )

    # Flexibility to generate single stratified dataset from a single file 
    # Just in generate_single_dataset
    if len(datafiles) > 1:
        train = ConcatDataset(train_datasets)
        val = ConcatDataset(val_datasets)
        test = ConcatDataset(test_datasets)
    else:
        train = train_datasets[0]
        val = val_datasets[0]
        test = test_datasets[0]

    return train, val, test

def generate_single_dataset(
    datafile: str,
    labelfile: str,
    class_label: str,
    test_prop=0.2,
    *args,
    **kwargs,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Generate a train/test split for the given datafile and labelfile.

    Parameters:
    datafile: Path to dataset csv file
    labelfile: Path to label csv file 
    class_label: Column (label) in labelfile to train on 

    Returns:
    Tuple[Dataset, Dataset]: Train/val/test set, respectively 
    """

    train, val, test = generate_datasets(
        datafiles=[datafile],
        labelfiles=[labelfile],
        class_label=class_label,
        test_prop=test_prop,
        *args,
        **kwargs,
    )

    return train, val, test 

def generate_single_dataloader(
    datafile: str,
    labelfile: str,
    class_label: str,
    test_prop=0.2,
    batch_size=4,
    num_workers=0,
    *args,
    **kwargs,
) -> DataLoader:
    train, val, test = generate_single_dataset(
        datafile=datafile,
        labelfile=labelfile,
        class_label=class_label,
        test_prop=test_prop,
        *args,
        **kwargs 
    )

    train = DataLoader(train, batch_size=batch_size, num_workers=num_workers)
    val = DataLoader(val, batch_size=batch_size, num_workers=num_workers)
    test = DataLoader(test, batch_size=batch_size, num_workers=num_workers)

    return train, val, test 

def generate_loaders(
    datafiles: List[str], 
    labelfiles: List[str],
    class_label: str,
    batch_size: int=4, 
    num_workers: int=0,
    collocate: bool=False, 
    *args,
    **kwargs,
) -> Union[Tuple[List[DataLoader], List[DataLoader], List[DataLoader]], Tuple[DataLoader, DataLoader, DataLoader]]:

    if not collocate:
        train, val, test = [], [], []
        for datafile, labelfile in zip(datafiles, labelfiles):
            trainloader, valloader, testloader = generate_single_dataloader(
                datafile=datafile,
                labelfile=labelfile,
                class_label=class_label, 
                batch_size=batch_size,
                num_workers=num_workers,
                *args,
                **kwargs
            )

            train.append(trainloader)
            val.append(valloader)
            test.append(testloader)
    else:
        train, val, test = generate_datasets(
            datafiles=datafiles,
            labelfiles=labelfiles,
            class_label=class_label,
            *args,
            **kwargs,
        )

        train = DataLoader(train, batch_size=batch_size, num_workers=num_workers)
        val = DataLoader(train, batch_size=batch_size, num_workers=num_workers)
        test = DataLoader(train, batch_size=batch_size, num_workers=num_workers)

    return train, val, test 





