#!/bin/bash
#SBATCH --account=rrg-dwishart
#SBATCH --nodes=1
#SBATCH --job-name=deepmet_pt7_frag_gen_arr
#SBATCH --mem=80G
#SBATCH --cpus-per-task=40
#SBATCH --time=0-1:00
#SBATCH --mail-user=fw4@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --output=/home/feiw/feiw/frag-gnn/jobs/deepmet_pt7_frag_gen_arr%j.out
#SBATCH --error=/home/feiw/feiw/frag-gnn/jobs/deepmet_pt7_frag_gen_arr%j.err
#SBATCH --open-mode=append
#SBATCH --array=1-3

# record start
echo `date`: Job ${SLURM_JOB_ID} is allocated resources
#load modules
module load StdEnv/2023
module load python/3.10
module load rdkit/2023.09.5

# go to script dir
cd /home/feiw/feiw/frag-gnn
# load env
source ./FRAG-GNN-GPU2/bin/activate

python preproc_scripts/deepmet_pcdl/02_preapre_deepmet_prediction_frag.py \
    --deepmet_moldf_fp=data/ms2c/hmdb/proc/mol_df_${SLURM_ARRAY_TASK_ID}.pkl.gz \
    --frag_dp=${SLURM_TMPDIR}/mol_df_${SLURM_ARRAY_TASK_ID} \
    --max_depth=3 

cd ${SLURM_TMPDIR}
tar -cf mol_df_${SLURM_ARRAY_TASK_ID}.tar mol_df_${SLURM_ARRAY_TASK_ID}
mkdir -p /scratch/feiw/frag-gnn-data/ms2c/deepmet/frags/pubchem_d3_h4/
cp mol_df_${SLURM_ARRAY_TASK_ID}.tar /scratch/feiw/frag-gnn-data/ms2c/deepmet/frags/pubchem_d3_h4/

# record ending
echo `date`: Job ${SLURM_JOB_ID} finished running

