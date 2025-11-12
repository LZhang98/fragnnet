#!/bin/bash

set -e

source ~/.bashrc

conda create -n FRAGNNET-GPU2
conda activate FRAGNNET-GPU2

conda install python=3.12

pip install  \
 torch==2.6.0 \
 torch_geometric==2.7.0 \
 rdkit==2024.09.6 \
 pandas==2.2.3 \
 numpy==1.26.4 \
 wandb \
 tqdm \
 scipy \
 joblib==1.3.2 \
 matplotlib \
 pytorch_lightning==2.5.0 \
 omegaconf \
 cython==3.0.12 \
 pyteomics \
 notebook \
 scikit-learn \
 h5py \
 numpy==1.26.4

 # dgl==2.4.0 -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html \
 # dgl==1.0.4 -f https://data.dgl.ai/wheels/repo.html \
pip install --no-deps -I -e .