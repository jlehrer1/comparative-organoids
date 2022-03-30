from ssl import Options
import linecache 
import csv
from typing import *
import random
from functools import cached_property

import comet_ml
import pandas as pd 
import torch
import numpy as np

from torch.utils.data import Dataset
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from torch import Tensor 

# Set all seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

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
        skip=2,
        cast=True,
        index_col='cell',
    ):
        self.filename = filename
        self.name = filename # alias 

        if indices is None:
            self._labeldf = pd.read_csv(labelname).set_index(index_col)
        else:
            self._labeldf = pd.read_csv(labelname).iloc[indices, :].set_index(index_col)

        self._total_data = 0
        self._class_label = class_label

        self.skip = skip
        self.cast = cast

    def __getitem__(self, idx):
        # The label dataframe contains both its natural integer index, as well as a "cell" index which contains the indices of the data that we 
        # haven't dropped. This is because some labels we don't want to use, i.e. the ones with "Exclude" or "Low Quality".
        # Since we are grabbing lines from a raw file, we have to keep the original indices of interest, even though the length
        # of the label dataframe is smaller than the original index
        idx = self._labeldf.iloc[idx].name
        
        # Get label
        label = self._labeldf.loc[idx, self._class_label]
        
        # get gene expression for current cell from csv file
        # We skip some lines because we're reading directly from 
        line = linecache.getline(self.filename, idx + self.skip)
        csv_data = csv.reader([line])
        data = [x for x in csv_data][0]
        
        if self.cast:
            data = torch.from_numpy(np.array([float(x) for x in data])).float()

        return data, label

    def __len__(self):
        return self._labeldf.shape[0] # number of total samples 
    
    def num_labels(self):
        return self._labeldf[self._class_label].nunique()
    
    def num_features(self):
        return len(self.__getitem__(0)[0])

    def getline(self, num):
        line = linecache.getline(self.filename, num)
        csv_data = csv.reader([line])
        data = [x for x in csv_data][0]
        
        return data 

    @cached_property # Worth caching, since this is a list comprehension on up to 50k strings. Annoying. 
    def features(self):
        data = self.getline(self.skip - 1)
        data = [x.upper().strip() for x in data]
             
        return data

    @property
    def columns(self): # Just an alias...
        return self.features

    @cached_property
    def labels(self):
        return self._labeldf.loc[:, self._class_label].unique()

    @property
    def shape(self):
        return (self.__len__, len(self.features))
        
def clean_sample(sample, refgenes, currgenes):
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

    return sample

def _dataset_class_weights(
    label_files: List[str],
    class_label: str,
) -> Tensor:
    """
    Compute class weights for the entire l  abel set of N labels.

    Parameters:
    label_files: List of absolute paths to label files

    Returns:
    np.array: Array of class weights for classes 0,...,N-1
    """

    comb = []

    for file in label_files:
        comb.extend(
            pd.read_csv(file).loc[:, class_label].values
        )

    return torch.from_numpy(compute_class_weight(
        classes=np.unique(comb),
        y=comb,
        class_weight='balanced',
    )).float()

def _generate_stratified_dataset(
    dataset_files: List[str], 
    label_files: List[str],
    class_label: str,
    skip=2,
    cast=True,
    test_prop: float=0.2,
    index_col='cell',
) -> Tuple[Dataset, Dataset]:
    """
    Generates train/val datasets with stratified label splitting. This means that the proportion of each class is the same 
    in the training and validation set. 
    
    Parameters:
    dataset_files: List of absolute paths to csv files under data_path/ that define cell x expression matrices
    label_files: List of absolute paths to csv files under data_path/ that define cell x class matrices
    class_label: Column in label files to train on. Must exist in all datasets, this should throw a natural error if it does not. 
    test_prop: Proportion of data to use as test set 

    Returns:
    Tuple[Dataset, Dataset]: Training and validation datasets, respectively
    """
    
    train_datasets = []
    val_datasets = []

    for datafile, labelfile in zip(dataset_files, label_files):
        # Read in current labelfile
        current_labels = pd.read_csv(labelfile).loc[:, class_label]
        
        # Make stratified split on labels
        trainsplit, valsplit = train_test_split(current_labels, stratify=current_labels, test_size=test_prop)
        
        # Generate train/test with stratified indices
        trainset = GeneExpressionData(
            filename=datafile, 
            labelname=labelfile,
            class_label=class_label,
            indices=trainsplit.index,
            skip=skip,
            cast=cast,
            index_col=index_col,
        )
        
        valset = GeneExpressionData(
            filename=datafile,
            labelname=labelfile,
            class_label=class_label,
            indices=valsplit.index, 
            skip=skip,
            cast=cast,
            index_col=index_col,
        )
        
        train_datasets.append(trainset)
        val_datasets.append(valset)
    
    train = torch.utils.data.ConcatDataset(train_datasets)
    val = torch.utils.data.ConcatDataset(val_datasets)

    return train, val

def _generate_split_dataset(
    dataset_files: List[str], 
    label_files: List[str],
    class_label: str,
    skip=2,
    cast=True,
    test_prop: float=0.2,
    index_col='cell',
) -> Tuple[Dataset, Dataset]:

    """
    Generates train/val datasets WITHOUT stratified splitting.
    
    Parameters:
    dataset_files: List of absolute paths to csv files under data_path/ that define cell x expression matrices
    label_files: List of absolute paths to csv files under data_path/ that define cell x class matrices
    class_label: Column in label files to train on. Must exist in all datasets, this should throw a natural error if it does not. 

    Returns:
    Tuple[Dataset, Dataset]: Training and validation datasets, respectively
    """

    datasets = []

    for datafile, labelfile in zip(dataset_files, label_files):
        subset = GeneExpressionData(
            filename=datafile,
            labelname=labelfile,
            class_label=class_label,
            skip=skip,
            cast=cast,
            index_col=index_col
        )

        datasets.append(subset)

    dataset = torch.utils.data.ConcatDataset(datasets)
    train_size = int((1. - test_prop) * len(dataset))
    test_size = len(dataset) - train_size
    train, test = torch.utils.data.random_split(dataset, [train_size, test_size])

    return train, test

def generate_datasets(
    dataset_files: List[str], 
    label_files: List[str],
    class_label: str,
    stratified=True,
) -> Tuple[Dataset, Dataset, int, int, Tensor]:
    """
    Generates the training / test set for the classifier, including input size and # of classes to be passed to the model object. 
    The assumption with all passed label files is that the number of classes in each dataset is the same. 
    Class labels are indexed from 0, so for N classes the labels are 0,...,N-1. 

    Parameters:
    dataset_files: List of absolute paths to csv files under data_path/ that define cell x expression matrices
    label_files: List of absolute paths to csv files under data_path/ that define cell x class matrices
    class_label: Column in label files to train on. Must exist in all datasets, this should throw a natural error if it does not. 
    stratified: bool=True: To return a stratified train/test split or not 

    Returns:
    Tuple[Dataset, Dataset, int, int, List[float]]: 
    Returns training dataset, validation dataset, input tensor size, number of class labels, class_weights respectively
    """

    if stratified:
        train, test = _generate_stratified_dataset(
            dataset_files=dataset_files,
            label_files=label_files,
            class_label=class_label,
        )
    else: 
        train, test = _generate_split_dataset(
            dataset_files=dataset_files,
            label_files=label_files,
            class_label=class_label,
        )

    # Calculate input tensor size and # of class labels
    input_size = len(train[0][0]) # Just consider the first sample for input shape
    num_labels = max(pd.read_csv(label_files[0]).loc[:, class_label].values) + 1 # Always N classes labeled 0,...,N-1

    return train, test, input_size, num_labels, _dataset_class_weights(label_files, class_label)
    