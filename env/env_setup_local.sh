#!/bin/bash

set -e

source ~/.bashrc

conda create -n FRAG-GNN
conda activate FRAG-GNN
conda install python=3.10

pip install torch==1.13.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu
pip install rdkit-pypi==2022.9.4
pip install cython
pip install pandas
pip install networkx
pip install wandb
pip install tqdm
pip install scipy
pip install joblib
pip install matplotlib
pip install pytest
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.0+cpu.html 
pip install dgl
pip install cysignals
pip install dgllife

pip install -I -e .