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
 torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cu121.html \
 pyg_lib \
 rdkit==2022.9.5 \
 pandas==2.2.3 \
 numpy==1.26.4 \
 wandb \
 tqdm \
 scipy \
 joblib==1.3.2 \
 matplotlib

pip install \
 cython==3.0.12 \
 pyteomics \
 pyopenms \
 notebook \
 scikit-learn \
 h5py

pip install --no-deps -I -e .