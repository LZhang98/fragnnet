#!/bin/bash

set -e

source ~/.bashrc

module load StdEnv/2023 rdkit/2023.09.5 python/3.11

virtualenv FRAG-GNN-GPU
source FRAG-GNN-GPU/bin/activate
pip install --no-index --upgrade pip

pip install --no-index torch==2.4.1
#pip install --no-index rdkit==2022.3.5
pip install --no-index torch_geometric==2.4.0 
pip install --no-index torch_scatter==2.1.2
pip install --no-index torch_sparse==0.6.18
pip install --no-index torch_cluster==1.6.3
pip install --no-index torch_spline_conv==1.2.2
pip install --no-index cython==3.0.11

pip install --no-index pandas
pip install --no-index networkx
pip install --no-index wandb
pip install --no-index tqdm
pip install --no-index scipy
pip install --no-index joblib==1.3.2
pip install --no-index matplotlib
pip install --no-index pytest
pip install --no-index pytorch_lightning==2.1.2
pip install --no-index CairoSVG
pip install --no-index dgl==2.4.0
pip install --no-index omegaconf

pip install cysignals
pip install pyteomics

# this is only needed on CC because there is no internet 
# pip install wandb[sweeps]
pip install -I -e .