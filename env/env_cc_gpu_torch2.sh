#!/bin/bash

set -e

source ~/.bashrc

module load python/3.10
virtualenv --no-download FRAG-GNN-GPU2
source FRAG-GNN-GPU2/bin/activate
pip install --no-index --upgrade pip

pip install torch==2.0.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install rdkit-pypi==2022.9.4
pip install torch_geometric pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
pip install cython
pip install cysignals
pip install pandas
pip install networkx
pip install wandb
pip install tqdm
pip install scipy
pip install joblib
pip install matplotlib
pip install pytest
pip install dash
pip install jupyter-dash
pip install CairoSVG
pip install lightning

pip install -I -e .