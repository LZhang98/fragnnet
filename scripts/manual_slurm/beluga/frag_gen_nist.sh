#!/bin/bash
#SBATCH --account=rrg-dwishart
#SBATCH --nodes=1
#SBATCH --job-name=frag_gen
#SBATCH --mem=90G
#SBATCH --cpus-per-task=40
#SBATCH --time=0-2:00
#SBATCH --mail-user=fw4@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --output=/home/feiw/feiw/frag-gnn/jobs/frag_gen_frag_gen_%j.out
#SBATCH --error=/home/feiw/feiw/frag-gnn/jobs/frag_gen_frag_gen_%j.err
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

DATASET_NAME = nist20mona23v3
MAX_DEPTH = 3

# create local dir
mkdir -p ${SLURM_TMPDIR}/proc
mkdir -p ${SLURM_TMPDIR}/frag/${DATASET_NAME}
# copy file to slrum local dir

cp -r ./data/proc/${DATASET_NAME} ${SLURM_TMPDIR}/proc/

python preproc_scripts/03_prepare_dag_feats.py \
    --frag_dp=${SLURM_TMPDIR}/frag/${DATASET_NAME} \
    --proc_dp=./data/proc/${DATASET_NAME}  \
    --max_depth=${MAX_DEPTH} \
    --wandb_mode=off

# go to frag directory
cd ${SLURM_TMPDIR}/frag/

tar -cf ${DATASET_NAME}.tar ${DATASET_NAME}
mkdir -p /scratch/feiw/frag-gnn-data/frags/d${MAX_DEPTH}_h4_isoFalse/
cp ${DATASET_NAME}.tar /scratch/feiw/frag-gnn-data/frags/d${MAX_DEPTH}_h4_isoFalse/

# record ending
echo `date`: Job ${SLURM_JOB_ID} finished running
