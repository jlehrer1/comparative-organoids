# *CerebalCell*: A Deep Learning Model for Classifying Cortical Cells

This codebase serves as the main repository for my work at the [One Brain](https://mostajo-radji.com/) group at the UCSC Genomics Insitute, where I'm using deep learning to classifying single cells from RNA-seq expression data. 

The concept is simple: we want to understand the distribution of cells in brain organoids grown from various cell lines, and how accurately they model the cortical tissue in a real human brain. So, we train a model to classify cells into their cortical subtype from human tissue, then use this trained model to classify organoid cells. This will give us a quantitative way to compare organoid protocols, in our mission to understand the cell makeup and function of the human cortex. 

The project is organized into four essential parts:

1. UMAP for dimensionality reduction
2. HDBSCAN for non-parametric clustering (well, non-parametric in the sense of not knowing the number of cluster a-priori)
3. Manual and biological knowledge based cluster annotation
4. Training a deep learning model on the human tissue, and classifying the organoid cells

The project is structured in the following way:
The `src/` directory contains all code for steps 1-4 above. `src/data` curates and cleans the human tissue data, and uploads/downloads to our S3 bucket as needed. `src/models` contains the code for dimensionality reduction, clustering, and model training. Finally, the top level folder `yaml` contains the Kubernetes yaml files for running model/dimensionality reduction/clustering code on our compute cluster, the [Pacific Research Platform](https://pacificresearchplatform.org/). 

The project is currently at step 3, where we're using our current biological knowledge of gene expression to classify human tissue cell clusters. 