FROM pytorch/pytorch

WORKDIR /src

RUN apt-get update
RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN apt-get --allow-releaseinfo-change update && \
    apt-get install -y --no-install-recommends \
        curl \
        sudo \
        vim 

RUN conda install --yes boto3 tenacity pandas numpy pip plotly scipy && \
    conda install -c conda-forge python-kaleido dask-xgboost hdbscan dask-xgboost && \
    pip install matplotlib && \
    pip install umap-learn && \
    pip install dask && \
    pip install dask-ml && \
    pip install pynndescent && \ 
    pip install seaborn && \
    pip install imbalanced-learn && \ 
    pip install xgboost && \ 
    pip install pytorch-lightning && \ 
    pip install comet_ml && \ 
    pip install wandb && \ 
    pip install transposecsv==0.0.5

COPY . .