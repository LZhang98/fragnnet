#!/bin/bash

set -e


source ~/.bashrc

conda create -n FRAG-GNN-GPU
conda activate FRAG-GNN-GPU

conda install python=3.10

pip install torch==2.2.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install rdkit-pypi==2022.9.5
pip install torch_geometric==2.5 torch_scatter -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
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
pip install dgl==1.0.4 -f https://data.dgl.ai/wheels/cu118/repo.html
pip install omegaconf
pip install pyteomics

pip install -I -e .