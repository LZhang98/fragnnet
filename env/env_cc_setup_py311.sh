#!/bin/bash

set -e

source ~/.bashrc

module load StdEnv/2023
module load python/3.11

virtualenv FRAGNNET
source FRAGNNET/bin/activate
pip install --no-index --upgrade pip

pip install --no-index \
  torch==2.4.1 \
  torch_geometric==2.6.1 \
  torch_scatter==2.1.2 \
  pandas \
  wandb \
  tqdm \
  scipy==1.15.2 \
  joblib==1.3.2 \
  matplotlib \
  lightning==2.4.0 \
  omegaconf \
  cython==3.0.12 \
  pyopenms==3.3.0 \
  pandarallel==1.6.5 \
  h5py

pip install pyteomics torch-linear-assignment

pip install --no-deps -I -e .