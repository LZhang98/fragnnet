#!/bin/bash

set -e

source ~/.bashrc

module load StdEnv/2020
module load python/3.10
module load cuda/11.7
module load rust/1.70.0

virtualenv FRAG-GNN-GPU2
source FRAG-GNN-GPU2/bin/activate
pip install --no-index --upgrade pip

pip install --no-index torch==2.0.1
pip install --no-index rdkit==2022.3.5
pip install --no-index torch_geometric==2.3.1 
pip install --no-index torch_scatter==2.1.1
pip install --no-index torch_sparse==0.6.17
pip install --no-index torch_cluster==1.6.1
pip install --no-index torch_spline_conv==1.2.2
#pip install pyg_lib -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
pip install --no-index cython
pip install cysignals
pip install --no-index pandas
pip install --no-index networkx
pip install --no-index wandb
pip install --no-index tqdm
pip install --no-index scipy
pip install --no-index joblib
pip install --no-index matplotlib
pip install --no-index pytest
pip install --no-index pytorch_lightning==2.0.2
pip install CairoSVG
pip install --no-index dgl==1.1.1
# this is only needed on CC because there is no internet 
# pip install wandb[sweeps]
pip install -I -e .