from ssl import Options
import linecache 
import csv
from typing import *
import random

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

    Parameters:
    filename: Path to csv data file, where rows are samples and columns are features
    labelname: Path to label file, where column '# labels' defines classification labels
    """

    def __init__(
        self, 
        filename: str, 
        labelname: str, 
        class_label: str,
        indices: Union[np.array, List[int]]=None
    ):
        
        self._filename = filename
        self._labelname = (pd.read_csv(labelname) if indices is None else pd.read_csv(labelname).iloc[indices, :])
        self._total_data = 0
        self._class_label = class_label
        self.index = indices
        
        with open(filename, "r") as f:
            self._total_data = len(self.index)
            
    def __getitem__(self, idx):
        # Get index in dataframe from integer index
        idx = self._labelname.iloc[idx].name
        
        # Get label
        label = self._labelname.loc[idx, self._class_label]
        
        # get gene expression for current cell from csv file
        # index + 2: Header is blank line and zeroth row is column names
        line = linecache.getline(self._filename, idx + 2)
        csv_data = csv.reader([line])
        data = [x for x in csv_data][0]
        
        return torch.from_numpy(np.array([float(x) for x in data])).float(), label
    
    def __len__(self):
        return self._total_data
    
    def num_labels(self):
        return self._labelname[self._class_label].nunique()
    
    def num_features(self):
        return len(self.__getitem__(0)[0])

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

    comb = np.array([pd.read_csv(file).loc[:, class_label].values for file in label_files]).flatten()

    return torch.from_numpy(compute_class_weight(
        classes=np.unique(comb),
        y=comb,
        class_weight='balanced',
    )).float()

def _generate_stratified_dataset(
    dataset_files: List[str], 
    label_files: List[str],
    class_label:str,
) -> Tuple[Dataset, Dataset]:
    """Generates train/val datasets with stratified label splitting. This means that the proportion of each class is the same 
    in the training and validation set. 
    
    Parameters:
    dataset_files: List of absolute paths to csv files under data_path/ that define cell x expression matrices
    label_files: List of absolute paths to csv files under data_path/ that define cell x class matrices
    class_label: Column in label files to train on. Must exist in all datasets, this should throw a natural error if it does not. 

    Returns:
    Tuple[Dataset, Dataset]: Training and validation datasets, respectively
    """
    
    train_datasets = []
    val_datasets = []

    for datafile, labelfile in zip(dataset_files, label_files):
        # Read in current labelfile
        current_labels = pd.read_csv(labelfile).loc[:, class_label]
        
        # Make stratified split on labels
        trainsplit, valsplit = train_test_split(current_labels, stratify=current_labels, test_size=0.8)
        
        # Generate train/test with stratified indices
        trainset = GeneExpressionData(
            filename=datafile, 
            labelname=labelfile,
            class_label='Subtype',
            indices=trainsplit.index 
        )
        
        valset = GeneExpressionData(
            filename=datafile, 
            labelname=labelfile,
            class_label='Subtype',
            indices=valsplit.index 
        )
        
        train_datasets.append(trainset)
        val_datasets.append(valset)
    
    train = torch.utils.data.ConcatDataset(train_datasets)
    val = torch.utils.data.ConcatDataset(val_datasets)

    return train, val 

def _generate_split_dataset(
    dataset_files: List[str], 
    label_files: List[str],
    class_label:str,
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
            class_label=class_label
        )

        datasets.append(subset)

    dataset = torch.utils.data.ConcatDataset(datasets)
    train_size = int(0.80 * len(dataset))
    test_size = len(dataset) - train_size
    train, test = torch.utils.data.random_split(dataset, [train_size, test_size])

    return train, test 

def generate_datasets(
    dataset_files: List[str], 
    label_files: List[str],
    class_label:str,
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
    