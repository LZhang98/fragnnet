#!/bin/bash

set -e


source ~/.bashrc

conda create -n FRAG-GNN-GPU
conda activate FRAG-GNN-GPU

conda install -y python=3.10

pip install -q torch==2.2.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
pip install -q rdkit-pypi==2022.9.4
pip install -q torch_geometric==2.4.0 pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
pip install -q cython
pip install -q cysignals
pip install -q pandas
pip install -q networkx
pip install -q wandb
pip install -q tqdm
pip install -q scipy
pip install -q joblib
pip install -q matplotlib
pip install -q pytest
pip install -q CairoSVG
pip install -q lightning==2.1.2
pip install -q dgl==2.0.0 -f https://data.dgl.ai/wheels/cu121/repo.html
pip install -q omegaconf
pip install -q adjustText
pip install -q Pyarrow
pip install -q pyteomics

pip install -q -I -e .