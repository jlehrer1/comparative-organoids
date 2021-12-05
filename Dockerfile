FROM --platform=linux/amd64 continuumio/miniconda3 

WORKDIR /src

RUN apt-get update
RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN apt-get --allow-releaseinfo-change update && \
    apt-get install -y --no-install-recommends \
        glances \
        git \
        awscli \
        curl \
        ruby \
        sudo \
        vim \
        libxml-libxml-perl \ 
        time \
        ffmpeg \
        libsm6 \
        libxext6 \
        libgl1-mesa-glx \
        gzip \
        gawk \
        build-essential

RUN curl -s -L https://nvidia.github.io/nvidia-container-runtime/gpgkey | sudo apt-key add - \
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
    curl -s -L https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list |\
        sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list \
    sudo apt-get update \
    sudo apt-get install nvidia-container-runtime 

RUN conda install --yes boto3 tenacity pandas numpy pip plotly scipy 
RUN conda install -c conda-forge python-kaleido dask-xgboost hdbscan dask-xgboost 
RUN conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
RUN pip install statdepth==0.7.17 kaleido matplotlib umap-learn dask dask-ml pynndescent seaborn imbalanced-learn xgboost 

COPY . .