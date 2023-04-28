#!/bin/bash

set -e

source ~/.bashrc

conda create -n FRAG-GNN
conda activate FRAG-GNN
conda install python=3.10

pip install torch==2.0.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu
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
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-2.0.0+cpu.html
pip install dgl
pip install dgllife
# for flow chart
pip install dash
# for spectrum https://spectralentropy.readthedocs.io/en/master/#module-math_distance
pip install pip https://github.com/YuanyueLi/SpectralEntropy.git

pip install -I -e .