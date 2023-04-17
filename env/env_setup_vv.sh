#!/bin/bash

set -e

module load cuda-11.7

conda create -n FRAG-GNN
conda activate FRAG-GNN
conda install python=3.10

pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install rdkit-pypi==2022.9.4
pip install cython
pip install pandas
pip install networkx
pip install wandb
pip install tqdm
pip install scipy
pip install joblib
pip install matplotlib
pip install dgl -f https://data.dgl.ai/wheels/cu117/repo.html
pip install cysignals
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.13.1+cu117.html

pip install -I -e .