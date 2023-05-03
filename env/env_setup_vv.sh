#!/bin/bash

set -e

module load cuda-11.7

conda create -n FRAG-GNN
conda activate FRAG-GNN
conda install python=3.10

pip install torch==2.0.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install rdkit-pypi==2022.9.4
pip install torch_geometric pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
pip install dgl -f https://data.dgl.ai/wheels/cu117/repo.html
pip install dgllife
pip install cython
pip install pandas
pip install networkx
pip install wandb
pip install tqdm
pip install scipy
pip install joblib
pip install matplotlib
pip install pytest
pip install cysignals


pip install -I -e .