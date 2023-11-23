echo 'make sure run this at top level project dir'
source ~/.bashrc

module load StdEnv/2020
module load python/3.10
module load cuda/11.7
module load rust/1.70.0

source FRAG-GNN-GPU2/bin/activate

#export WANDB_DIR='./wandb'
#export WANDB_CACHE_DIR='./wandb/.cache'
#export WANDB_CONFIG_DIR='./wandb/.config'

# https://github.com/wandb/wandb/issues/2423
wandb sync --sync-all --append
