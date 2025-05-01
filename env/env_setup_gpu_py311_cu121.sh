#!/bin/bash

set -e

source ~/.bashrc

conda create -n FRAG-GNN-GPU
conda activate FRAG-GNN-GPU

conda install python=3.11

pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric==2.6.1
pip install pyg_lib torch_scatter -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
pip install rdkit==2024.3.5
pip install pandas
pip install wandb
pip install tqdm
pip install scipy
pip install joblib==1.3.2
pip install matplotlib
pip install pytorch_lightning==2.4.0
pip install torchdata==0.8.0
pip install dgl==2.4.0 -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html 
pip install omegaconf
pip install cython==3.0.12
pip install omegaconf
pip install pyteomics
pip install notebook

pip install -I -e .