import linecache
from multiprocessing.sharedctypes import Value 
from typing import *
from functools import cached_property, partial
from itertools import chain 
import inspect

import pandas as pd 
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import pytorch_lightning as pl 

import sys, os 
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

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
        sep=',',
        **kwargs, # To handle extraneous inputs 
    ):
        self.filename = filename
        self.labelname = labelname # alias 
        self.class_label = class_label
        self.index_col = index_col 
        self.skip = skip
        self.cast = cast
        self.indices = indices
        self.sep = sep

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

        # The actual line in the datafile to get, corresponding to the number in the self.index_col values 
        data_index = self._labeldf.loc[idx, self.index_col]

        # get gene expression for current cell from csv file
        # We skip some lines because we're reading directly from 
        line = linecache.getline(self.filename, data_index + self.skip)
        
        if self.cast:
            data = torch.from_numpy(np.array(line.split(self.sep), dtype=np.float32)).float()
        else:
            data = np.array(line.split(self.sep))

        label = self._labeldf.loc[idx, self.class_label]

        return data, label

    def __len__(self):
        return len(self._labeldf) # number of total samples 

    @cached_property
    def columns(self): # Just an alias...
        return self.features

    @cached_property # Worth caching, since this is a list comprehension on up to 50k strings. Annoying. 
    def features(self):
        data = linecache.getline(self.filename, self.skip - 1)
        data = [x.split('|')[0].upper().strip() for x in data.split(self.sep)]

        return data

    @cached_property
    def labels(self):
        return self._labeldf.loc[:, self.class_label].unique()

    @property
    def shape(self):
        return (self.__len__(), len(self.features))
    
    @cached_property 
    def class_weights(self):
        labels = self._labeldf.loc[:, self.class_label].values

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
            f"indices={self.indices})"
        )

def _collate_with_refgenes(
    sample: List[tuple], 
    refgenes: List[str], 
    currgenes: List[str],
    transpose: bool,
    normalize: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate minibatch of samples where we're intersecting the columns between refgenes and currgenes,
    optionally normalizing and transposing.

    Parameters:
    sample: List of samples from GeneExpressionData object
    refgenes: List of reference genes
    currgenes: List of current columns from sample 
    transpose: boolean, indicates if we should transpose the minibatch (in the case of incorrectly formatted .csv data)
    normalize: boolean, indicates if we should normalize the minibatch

    Returns:
    Two torch.Tensors containing the data and labels, respectively
    """

    data = clean_sample(torch.stack([x[0] for x in sample]), refgenes, currgenes)
    labels = torch.tensor([x[1] for x in sample])

    return _transform_sample(data, normalize, transpose), labels 

def _standard_collate(
    sample: List[tuple],
    normalize: bool,
    transpose: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate minibatch of samples, optionally normalizing and transposing. 

    Parameters:
    sample: List of GeneExpressionData items to collate
    transpose: boolean, indicates if we should transpose the minibatch (in the case of incorrectly formatted .csv data)
    normalize: boolean, indicates if we should normalize the minibatch

    Returns:
    Two torch.Tensors containing the data and labels, respectively
    """

    data = torch.stack([x[0] for x in sample])
    labels = torch.tensor([x[1] for x in sample])

    return _transform_sample(data, normalize, transpose), labels 

def _transform_sample(
    data: torch.Tensor, 
    normalize: bool, 
    transpose: bool
) -> torch.Tensor:
    """
    Optionally normalize and tranpose a torch.Tensor
    """
    if transpose:
        data = data.T

    if normalize:
        data = torch.nn.functional.normalize(data)

    return data 

class CollateLoader(DataLoader):
    """
    Subclass of DataLoader that creates a collate_fn on the fly as required by the user. This is used in the case where we want to calculate the intersection
    between a sample and the total column intersection of our data. We do this batch-wise instead of sample-wise for speed, since numpy is efficient at working on 2d arrays.

    Parameters:
    dataset: GeneExpressionDataset to create DataLoader from
    refgenes: Optional, list of columns to take intersection with 
    currgenes: Optional, list of current dataset columns
    transpose: Boolean indicating whether to tranpose the batch data 
    normalize: Boolean indicating whether to normalize the batch data 
    """
    def __init__(
        self, 
        dataset: GeneExpressionData,
        refgenes: List[str]=None, 
        currgenes: List[str]=None, 
        transpose: bool=False, 
        normalize: bool=False,
        **kwargs,
    ) -> None:

        if refgenes is None and currgenes is not None or refgenes is not None and currgenes is None:
            raise ValueError("If refgenes is passed, currgenes must be passed too. If currgenes is passed, refgenes must be passed too.")
        
        # Create collate_fn via a partial of the possible collators, depending on if columns intersection is being calculated
        if refgenes is not None:
            collate_fn = partial(_collate_with_refgenes, refgenes=refgenes, currgenes=currgenes, transpose=transpose, normalize=normalize)
        else:
            collate_fn = partial(_standard_collate, normalize=normalize, transpose=transpose)

        allowed_args = inspect.signature(super().__init__).parameters
        new_kwargs = {}

        # This is awkward, but Dataloader init doesn't handle optional keyword arguments
        # So we have to take the intersection between the passed **kwargs and the DataLoader named arguments
        for key in allowed_args:
            name = allowed_args[key].name
            if name in kwargs:
                new_kwargs[key] = kwargs[key]

        super().__init__(
            dataset=dataset,
            collate_fn=collate_fn, 
            **new_kwargs,
        )

class SequentialLoader:
    """
    Class to sequentially stream samples from an arbitrary number of DataLoaders.

    Parameters:
    dataloaders: List of DataLoaders or DataLoader derived class, such as the CollateLoader from above 
    """
    def __init__(self, dataloaders):
        self.dataloaders = dataloaders

    def __len__(self):
        return sum([len(dl) for dl in self.dataloaders])

    def __iter__(self):
        yield from chain(*self.dataloaders)

def clean_sample(
    sample: torch.Tensor, 
    refgenes: List[str],
    currgenes: List[str],
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
    combine=False,
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
    
    if combine and len(datafiles) == 1:
        raise ValueError('Cannot combine datasets when number of datafiles == 1.')

    if len(datafiles) == 1:
        return generate_single_dataset(
            datafiles[0],
            labelfiles[0],
            class_label,
            test_prop,
            **kwargs,
        )

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
                    **kwargs,
                )
            )

    # Flexibility to generate single stratified dataset from a single file 
    # Just in generate_single_dataset
    if combine:
        train_datasets = ConcatDataset(train_datasets)
        val_datasets = ConcatDataset(val_datasets)
        test_datasets = ConcatDataset(test_datasets)

    return train_datasets, val_datasets, test_datasets

def generate_single_dataset(
    datafile: str,
    labelfile: str,
    class_label: str,
    test_prop=0.2,
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
    current_labels = pd.read_csv(labelfile).loc[:, class_label]
    
    # Make stratified split on labels
    trainsplit, valsplit = train_test_split(current_labels, stratify=current_labels, test_size=test_prop)
    trainsplit, testsplit = train_test_split(trainsplit, stratify=trainsplit, test_size=test_prop)

    train, val, test = (
        GeneExpressionData(
            filename=datafile,
            labelname=labelfile,
            class_label=class_label,
            indices=indices,
            **kwargs,
        )
        for indices in [trainsplit, valsplit, testsplit]  
    )

    return train, val, test 

def generate_single_dataloader(
    **kwargs,
) -> Tuple[CollateLoader, CollateLoader, CollateLoader]:

    train, val, test = generate_single_dataset(
        **kwargs,
    )

    loaders = (
        CollateLoader(
                dataset=dataset, 
                currgenes=(dataset.columns if 'refgenes' in kwargs.keys() else None),
                **kwargs,
            )
        for dataset in [train, val, test]
    )

    return loaders 

def generate_dataloaders(
    datafiles: List[str], 
    labelfiles: List[str],
    collocate: bool=True, 
    **kwargs,
) -> Union[Tuple[List[CollateLoader], List[CollateLoader], List[CollateLoader]], Tuple[SequentialLoader, SequentialLoader, SequentialLoader]]:

    if len(datafiles) != len(labelfiles):
        raise ValueError("Must have same number of datafiles and labelfiles")
    
    if collocate and len(datafiles) == 1:
        raise ValueError("Cannot collocate dataloaders with only one dataset file")

    train, val, test = [], [], []
    for datafile, labelfile in zip(datafiles, labelfiles):
        trainloader, valloader, testloader = generate_single_dataloader(
            datafile=datafile,
            labelfile=labelfile,
            **kwargs,
        )

        train.append(trainloader)
        val.append(valloader)
        test.append(testloader)

    if len(datafiles) == 1:
        train = train[0]
        val = val[0]
        test = test[0]

    if collocate: # Join these together into sequential loader if requested  
        train, val, test = SequentialLoader(train), SequentialLoader(val), SequentialLoader(test)

    return train, val, test 

        





