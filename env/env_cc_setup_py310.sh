#!/bin/bash

set -e

source ~/.bashrc

module load StdEnv/2023
module load python/3.10

virtualenv FRAGNNET-GPU
source FRAGNNET-GPU/bin/activate
pip install --no-index --upgrade pip

pip install --no-index \
  torch==2.4.1 \
  torch_geometric==2.4.0 \
  torch_scatter==2.1.2 \
  pandas \
  wandb \
  tqdm \
  scipy \
  joblib==1.3.2 \
  matplotlib \
  lightning==2.1.2 \
  omegaconf \
  cython==3.0.8 \
  pyopenms==3.3.0 \
  numpy==1.26.4 \
  pandarallel==1.6.5 \
  h5py

pip install pyteomics torch-linear-assignment

pip install --no-deps -I -e .