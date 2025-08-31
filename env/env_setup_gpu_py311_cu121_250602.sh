#!/bin/bash

set -e

source ~/.bashrc

conda create -n FRAGNNET-GPU
conda activate FRAGNNET-GPU

conda install python=3.11

pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric==2.6.1
pip install pyg_lib torch_scatter -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
pip install rdkit==2022.9.5
pip install pandas==2.2.3
pip install numpy==1.26.4
pip install wandb
pip install tqdm
pip install scipy
pip install joblib==1.3.2
pip install matplotlib
pip install pytorch_lightning==2.4.0
#pip install dgl==2.4.0 -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html 
pip install dgl==1.0.4 -f https://data.dgl.ai/wheels/repo.html
pip install omegaconf
pip install cython==3.0.12
pip install pyteomics
pip install notebook
pip install scikit-learn
pip install h5py
pip install --no-index numpy==1.26.4

pip install --no-deps -I -e .