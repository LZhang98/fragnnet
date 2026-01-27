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
  dgl==2.1.0 \
  omegaconf \
  cython==3.0.12 \
  h5py

pip install pyteomics

pip install --no-deps -I -e .