# *CerebalCell*: A Deep Learning Model for Classifying Cortical Cells

This codebase serves as the main repository for my work at the [One Brain](https://mostajo-radji.com/) group at the UCSC Genomics Insitute, where I'm using deep learning to classifying single cells from RNA-seq expression data. 

The concept is simple: we want to understand the distribution of cells in brain organoids grown from various cell lines, and how accurately they model the cortical tissue in a real human brain. So, we train a model to classify cells into their cortical subtype from human tissue, then use this trained model to classify organoid cells. This will give us a quantitative way to compare organoid protocols, in our mission to understand the cell makeup and function of the human cortex. 

All source code is in `src/`.
### `src/data/`
This folder holds all scripts and methods for downloading, unzipping, renaming, and transposing the raw expression matrices from all dataset. 

`data_methods.py`: Contains source methods for combining and cleaning all external datasets.  
`clean_data.py`: Contains script for combining and cleaning all external datasets to be label-consistent and column order consistent for model training.  
`download_data.py`: Contains source methods for downloading and unzipping the raw expression matrices from cells.ucsc.edu, as well as methods for downloading data from the braingeneersdev S3 bucket, once the data is processed and uploaded there.  
`transpose_data.py`: Since all expression matrices are `gene x cell` and we want to train on `cell x gene` so we can classify individual cells, this script calculates the transpose of all `data/raw` data and writes it to `data/interim`. This is nontrivial, and uses my library [transposecsv](https://github.com/jlehrer1/transpose-csv).  

### `src/features`
This folder holds all scripts related to feature generation, data augmentation, and label encoding.

### `src/models`
This folder contains all scripts and methods for defining our neural network model, defining our dataset, training our model and deploying it to the Nautilus cluster for distributed training.  

`lib/data.py`: The PyTorch dataset for reading in csv files too big to fit in memory, as well as methods that generate the train, test and validation split across an arbitrary number of data and label files.  
`lib/neural.py`: The PyTorch Lightning NN architecture. 
`train_neural_network.py`: Trains the NN model locally.
`run_model_search.py`: Sets up an fixed number of jobs on the Nautilus cluster training the NN model on GPU, with randomly initialized hyperparameters.

### `src/visualization`
`visualize.py`: Contains code for UMAP plot generation.
