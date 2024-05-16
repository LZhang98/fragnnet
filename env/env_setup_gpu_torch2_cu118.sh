#!/bin/bash

set -e


source ~/.bashrc

conda create -n FRAG-GNN-GPU3
conda activate FRAG-GNN-GPU3

conda install python=3.10

pip install torch==2.0.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install rdkit-pypi==2022.9.4
pip install torch_geometric pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install cython
pip install cysignals
pip install pandas
pip install networkx
pip install wandb
pip install tqdm
pip install scipy
pip install joblib==1.3.2
pip install matplotlib
pip install pytest
pip install CairoSVG
pip install lightning
pip install dgl==1.0.4 -f https://data.dgl.ai/wheels/cu118/repo.html
pip install omegaconf

pip install -I -e .