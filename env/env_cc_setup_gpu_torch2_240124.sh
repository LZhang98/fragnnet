#!/bin/bash

set -e

source ~/.bashrc

module load StdEnv/2023
module load python/3.10
module load rust/1.70.0

virtualenv FRAG-GNN
source FRAG-GNN/bin/activate
pip install --no-index --upgrade pip

pip install --no-index torch==2.1.2
pip install --no-index rdkit==2022.3.5
pip install --no-index torch_geometric==2.4.0 
pip install --no-index torch_scatter==2.1.2
pip install --no-index torch_sparse==0.6.18
pip install --no-index torch_cluster==1.6.3
pip install --no-index torch_spline_conv==1.2.2
pip install --no-index pytorch_lightning==2.1.2
pip install --no-index dgl==2.0

pip install --no-index cython
pip install --no-index pandas
pip install --no-index networkx
pip install --no-index wandb
pip install --no-index tqdm
pip install --no-index scipy
pip install --no-index joblib==1.3.2
pip install --no-index matplotlib
pip install --no-index pytest

pip install cysignals

#pip install CairoSVG

pip install -I -e .