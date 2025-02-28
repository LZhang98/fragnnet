#!/bin/bash

set -e

source ~/.bashrc

module load StdEnv/2023 rdkit/2024.03.5 python/3.11

virtualenv FRAG-GNN-CPU
source FRAG-GNN-CPU/bin/activate
pip install --no-index --upgrade pip

pip install --no-index torch==2.4.1
#pip install --no-index rdkit==2022.3.5
pip install --no-index torch_geometric==2.6.1 
pip install --no-index torch_scatter
pip install --no-index torch_sparse
pip install --no-index torch_cluster
pip install --no-index torch_spline_conv
pip install --no-index cython==3.0.8
pip install cysignals
pip install --no-index pandas
pip install --no-index networkx
pip install --no-index wandb
pip install --no-index tqdm
pip install --no-index scipy
pip install --no-index joblib==1.3.2
pip install --no-index matplotlib
pip install --no-index pytest
pip install --no-index pytorch_lightning==2.4.0
pip install CairoSVG
pip install --no-index dgl==2.0.0
pip install omegaconf
pip install pyteomics

# this is only needed on CC because there is no internet 
# pip install wandb[sweeps]
pip install -I -e .