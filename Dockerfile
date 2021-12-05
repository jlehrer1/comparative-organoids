FROM pytorch/pytorch

WORKDIR /src

RUN apt-get update
RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN apt-get --allow-releaseinfo-change update && \
    apt-get install -y --no-install-recommends \
        curl \
        sudo \
        vim 

RUN conda install --yes boto3 tenacity pandas numpy pip plotly scipy 
RUN conda install -c conda-forge python-kaleido dask-xgboost hdbscan dask-xgboost 
RUN pip install matplotlib umap-learn dask dask-ml pynndescent seaborn imbalanced-learn xgboost 

COPY . .