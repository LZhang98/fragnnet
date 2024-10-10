#!/bin/bash
#SBATCH --account=def-rgreiner
#SBATCH --nodes=1
#SBATCH --job-name=ms2c_get_candidates_scaffold
#SBATCH --mem=96G
#SBATCH --cpus-per-task=40
#SBATCH --time=0-6:00
#SBATCH --mail-user=fw4@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --output=/home/feiw/feiw/frag-gnn/jobs/ms2c_get_candidates%j.out
#SBATCH --error=/home/feiw/feiw/frag-gnn/jobs/ms2c_get_candidates%j.err
#SBATCH --open-mode=append

# record start
echo `date`: Job ${SLURM_JOB_ID} is allocated resources
#load modules
module load StdEnv/2023
module load python/3.10

# go to script dir
cd /home/feiw/feiw/frag-gnn
# load env
source ./FRAG-GNN-GPU2/bin/activate

python preproc_scripts/pubchem_ms2c/02_prepare_ms2c_candidates.py  \
    --split_dp="./data/split/nist20mona23v3_d4_wr00_scaffold" \
    --proc_dp="./data/proc/nist20mona23v3" \
    --split_type=test

python preproc_scripts/pubchem_ms2c/03_perpare_ms2c_mol_and_spec_df.py \
	--candidate_pkl_fp=./data/ms2c/pubchem/candidates/pubchem_nist20mona23v3_d4_wr00_scaffold_test_10ppm_MORGAN-R2_50.pkl.gz \
	--custom_fp=config/nist20v3_d4_wr00_scaffold/fraggnn_d4/s0.yml

# record ending
echo `date`: Job ${SLURM_JOB_ID} finished running
