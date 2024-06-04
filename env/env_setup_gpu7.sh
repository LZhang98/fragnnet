#!/bin/bash

set -e


source ~/.bashrc

conda create -n FRAG-GNN-GPU5
conda activate FRAG-GNN-GPU5

conda install python=3.10

pip install torch==2.2.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
pip install rdkit-pypi==2022.9.5
pip install torch_geometric==2.5 torch_scatter -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
pip install cython
pip install cysignals
pip install pandas
pip install networkx
pip install wandb
# pip install tqdm
# pip install scipy
# pip install joblib
pip install matplotlib
pip install pytest
pip install CairoSVG
pip install lightning
pip install dgl -f https://data.dgl.ai/wheels/torch-2.2/cu121/repo.html
pip install pydantic
pip install omegaconf
pip install pyteomics

pip install -I -e .