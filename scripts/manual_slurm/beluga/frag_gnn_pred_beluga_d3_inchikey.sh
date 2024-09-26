#!/bin/bash
#SBATCH --account=rrg-dwishart
#SBATCH --nodes=1
#SBATCH --job-name=frag_gnn_pred_beluga_d3_inchikey
#SBATCH --mem=90G
#SBATCH --cpus-per-task=40
#SBATCH --time=0-12:00
#SBATCH --mail-user=fw4@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --output=/home/feiw/feiw/frag-gnn/jobs/frag_gnn_pred_beluga_d3_inchikey_%j.out
#SBATCH --error=/home/feiw/feiw/frag-gnn/jobs/frag_gnn_pred_beluga_d3_inchikey_%j.err
#SBATCH --open-mode=append


module load StdEnv/2023
module load cuda/11.7
module load python/3.10
module load rust/1.70.0

# record start
echo `date`: Job ${SLURM_JOB_ID} is allocated resources

cd ${SLURM_TMPDIR}

echo `date`: untar pubchem_nist20_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50 to  ${SLURM_TMPDIR}

cp /scratch/feiw/frag-gnn-data/ms2c/pubchem/frags/d3_h4_isoFalse/pubchem_nist20_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50.tar ./

tar -xf pubchem_nist20_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50.tar --checkpoint=.10000

echo `date`: running prediction

# go to script dir
cd /home/feiw/feiw/frag-gnn
# load env
source ./FRAG-GNN-GPU2/bin/activate

# pubchem_nist20_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50
python ./scripts/ms2c/run_frag_gnn_predict.py \
--frag_dp=${SLURM_TMPDIR}/pubchem_nist20_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50 \
--task_dp="data/ms2c/pubchem/pred_inputs/pubchem_nist20_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50" \
--custom_fp="config/nist20_d4_wr00_inchikey/fraggnn_d3_nodemlp_mnce_nocp.yml" \
--ckpt_fp="saved_ckpts/ohtiivyj_fraggnn_d3_nodemlp_mnce_nocp/ckpt/model-epoch=093.ckpt" \
--save_filename="ohtiivyj_fraggnn_d3_nodemlp_mnce_nocp.pkl.gz" \
--batch_size=1 \
--device="cpu"

# record start
echo `date`: Job ${SLURM_JOB_ID} is done