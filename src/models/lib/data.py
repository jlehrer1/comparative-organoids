from cProfile import label
from multiprocessing.sharedctypes import Value
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

from torch.utils.data import Dataset, DataLoader
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
        skip=3,
        cast=True,
        index_col='cell',
        normalize=False,
    ):
        self.filename = filename
        self.name = filename # alias 

        if indices is None:
            self._labeldf = pd.read_csv(labelname)
        else:
            self._labeldf = pd.read_csv(labelname).loc[indices, :]

        self._total_data = 0
        self._class_label = class_label
        self._index_col = index_col 

        self.skip = skip
        self.cast = cast
        self.normalize = normalize 

    def __getitem__(self, idx):
        # Get label
        if isinstance(idx, slice):
            if idx.start is None or idx.stop is None:
                raise ValueError(f"Error: Unlike other iterables, {self.__class__.__name__} does not support unbounded slicing since samples are being read as needed from disk, which may result in memory errors.")

            step = (1 if idx.step is None else idx.step)
            idxs = range(idx.start, idx.stop, step)

            return [self[i] for i in idxs]

        label = self._labeldf.loc[idx, self._class_label]

        # The label dataframe contains both its natural integer index, as well as a "cell" index which contains the indices of the data that we 
        # haven't dropped. This is because some labels we don't want to use, i.e. the ones with "Exclude" or "Low Quality".
        # Since we are grabbing lines from a raw file, we have to keep the original indices of interest, even though the length
        # of the label dataframe is smaller than the original index

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

        return data, label

    def __len__(self):
        return len(self._labeldf) # number of total samples 

    def getline(self, num):
        line = linecache.getline(self.filename, num)
        csv_data = csv.reader([line])
        data = [x for x in csv_data][0]
        
        return data 

    @property
    def columns(self): # Just an alias...
        return self.features

    @cached_property # Worth caching, since this is a list comprehension on up to 50k strings. Annoying. 
    def features(self):
        data = self.getline(self.skip - 1)
        data = [x.split('|')[0].upper().strip() for x in data]

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

# From: https://github.com/hcarlens/pytorch-tabular/blob/master/fast_tensor_data_loader.py
class FastTensorDataLoader(DataLoader):
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
            
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration

        batch = tuple(t[self.i: self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches

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

def _dataset_class_weights(
    label_files: List[str],
    class_label: str,
) -> Tensor:
    """
    Compute class weights for the entire label set of N labels.

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
    skip=3,
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
    skip=3,
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
            index_col=index_col,
        )

        datasets.append(subset)

    dataset = torch.utils.data.ConcatDataset(datasets)
    train_size = int((1. - test_prop) * len(dataset))
    test_size = len(dataset) - train_size
    train, test = torch.utils.data.random_split(dataset, [train_size, test_size])

    return train, test

def generate_single_dataset(
    datafile: str,
    labelfile: str,
    class_label: str,
    skip: int=2,
    index_col: str='cell', 
    cast: bool=True,
    test_prop=0.2,
    normalize=False,
) -> Tuple[Dataset, Dataset]:
    """
    Generate a train/test split for the given datafile and labelfile.

    Parameters:
    datafile: Path to dataset csv file
    labelfile: Path to label csv file 
    class_label: Column (label) in labelfile to train on 

    Returns:
    Tuple[Dataset, Dataset]: Train/val/test set, respectively 
    """

    dataset = GeneExpressionData(
        filename=datafile,
        labelname=labelfile,
        class_label=class_label,
        skip=skip,
        cast=cast,
        index_col=index_col,
        normalize=normalize,
    )

    # We have to do two splits for to generate train/test, then train --> train/val so we
    # can return train/val/test

    train_size = int((1. - test_prop) * len(dataset))
    test_size = len(dataset) - train_size
    train, test = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_size = int((1. - test_prop) * len(dataset))
    test_size = len(dataset) - train_size
    train, val = torch.utils.data.random_split(dataset, [train_size, test_size])

    return train, val, test 

def generate_datasets(
    dataset_files: List[str], 
    label_files: List[str],
    class_label: str,
    stratified=True,
    skip: int=2,
    index_col: str='cell', 
    cast: bool=True,
    test_prop=0.2
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
            skip=skip,
            cast=cast,
            index_col=index_col,
            test_prop=test_prop
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
    
def generate_single_dataloader(
    datafile: str,
    labelfile: str,
    class_label: str,
    skip: int=2,
    index_col: str='cell', 
    cast: bool=True,
    test_prop=0.2,
    normalize=False,
    batch_size=4,
    num_workers=0
) -> DataLoader:
    train, val, test = generate_single_dataset(
        datafile,
        labelfile,
        class_label,
        skip,
        index_col,
        cast,
        test_prop,
        normalize,
    )

    train = DataLoader(train, batch_size=batch_size, num_workers=num_workers)
    val = DataLoader(val, batch_size=batch_size, num_workers=num_workers)
    test = DataLoader(test, batch_size=batch_size, num_workers=num_workers)

    return train, val, test 

def generate_loaders(
    datafiles, 
    labelfiles,
    class_label,
    cast=True,
    skip=3,
    index_col='cell',
    normalize=False,
    batch_size=4, 
    num_workers=0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    trainloaders, valloaders, testloaders = [], [], []

    for datafile, labelfile in zip(datafiles, labelfiles):
        train, val, test = generate_single_dataloader(
            datafile=datafile,
            labelfile=labelfile,
            class_label=class_label, 
            skip=skip, 
            index_col=index_col, 
            cast=cast, 
            normalize=normalize,
        )

        trainloaders.append(train)
        valloaders.append(val)
        testloaders.append(test)

    return trainloaders, valloaders, testloaders
