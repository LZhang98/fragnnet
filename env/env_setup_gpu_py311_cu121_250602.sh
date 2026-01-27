#!/bin/bash

set -e

source ~/.bashrc

conda create -n FRAGNNET-GPU
conda activate FRAGNNET-GPU

conda install python=3.11

pip install \
 torch==2.4.1 \
lightning==2.4.0 \
 torch_geometric==2.6.1 \
 pyg_lib \
 torch_scatter \
 rdkit==2022.9.5 \
 pandas==2.2.3 \
 numpy==1.26.4 \
 wandb \
 tqdm \
 scipy \
 joblib==1.3.2 \
 matplotlib \

 dgl==1.1.3+cu121 \
 omegaconf \
 cython==3.0.12 \
 pyteomics \
 notebook \
 scikit-learn \
 h5py

pip install --no-deps -I -e .