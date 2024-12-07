#!/bin/bash
#SBATCH --account=def-rgreiner
#SBATCH --nodes=1
#SBATCH --job-name=frag_gen_pubchem_inchikey_d3
#SBATCH --mem=128G
#SBATCH --cpus-per-task=64
#SBATCH --time=0-3:00
#SBATCH --mail-user=fw4@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --output=/lustre06/project/6033476/feiw/frag-gnn/jobs/frag_gen_pubchem_inchikey_d3%j.out
#SBATCH --error=/lustre06/project/6033476/feiw/frag-gnn/jobs/frag_gen_pubchem_inchikey_d3%j.err
#SBATCH --open-mode=append

# record start
echo `date`: Job ${SLURM_JOB_ID} is allocated resources
#load modules
module load StdEnv/2023
module load python/3.10
module load rdkit/2023.09.5

# go to script dir
cd /lustre06/project/6033476/feiw/frag-gnn
# load env
source ./FRAG-GNN-GPU2/bin/activate

python preproc_scripts/08_prepare_ms2c_candidates.py  \
    --split_dp="./data/split/nist20_d4_wr00_inchikey" \
    --split_type=test 

python preproc_scripts/08_prepare_ms2c_candidates.py  \
    --split_dp="./data/split/nist20_d4_wr00_scaffold" \
    --split_type=test

# record ending
echo `date`: Job ${SLURM_JOB_ID} finished running
