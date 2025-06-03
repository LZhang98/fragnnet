#!/bin/bash

set -e

source ~/.bashrc

module load StdEnv/2023
module load python/3.11
module load rdkit/2023.09.5

virtualenv FRAG-GNN
source FRAG-GNN/bin/activate
pip install --no-index --upgrade pip

pip install --no-index torch==2.4.1
pip install --no-index torch_geometric==2.6.1 
pip install --no-index torch_scatter=2.1.2
pip install --no-index pandas==2.2.3
pip install --no-index wandb
pip install --no-index tqdm
pip install --no-index scipy
pip install --no-index joblib==1.3.2
pip install --no-index matplotlib
pip install --no-index pytorch_lightning==2.4.0
pip install --no-index dgl==2.4.0
pip install --no-index omegaconf
pip install --no-index cython==3.0.12

#pip install CairoSVG

pip install -I -e .