#!/bin/bash
#SBATCH --account=rrg-dwishart
#SBATCH --nodes=1
#SBATCH --job-name=frag_gnn_pred_beluga
#SBATCH --mem=90G
#SBATCH --cpus-per-task=40
#SBATCH --time=0-1:00
#SBATCH --mail-user=fw4@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --output=/home/feiw/feiw/frag-gnn/jobs/frag_gnn_pred_beluga_%j.out
#SBATCH --error=/home/feiw/feiw/frag-gnn/jobs/frag_gnn_pred_beluga_%j.err
#SBATCH --open-mode=append

module load StdEnv/2023
module load python/3.10

# record start
echo `date`: Job ${SLURM_JOB_ID} is allocated resources

cd ${SLURM_TMPDIR}

echo `date`: untar nist20_d4_wr00_inchikey_test_dags to  ${SLURM_TMPDIR}

cp /scratch/feiw/frag-gnn/data/ms2c/pubchem/frags/nist20_d4_wr00_inchikey_test_dags.tar ./
mkdir nist20_d4_wr00_inchikey_test_dags
tar -xf nist20_d4_wr00_inchikey_test_dags.tar -C nist20_d4_wr00_inchikey_test_dags  --checkpoint=.10000


echo `date`: untar nist20_d4_wr00_scaffold_test_dags to  ${SLURM_TMPDIR}

cp /scratch/feiw/frag-gnn/data/ms2c/pubchem/frags/nist20_d4_wr00_scaffold_test_dags.tar ./
mkdir nist20_d4_wr00_scaffold_test_dags
tar -xf nist20_d4_wr00_scaffold_test_dags.tar -C nist20_d4_wr00_scaffold_test_dags  --checkpoint=.10000

echo `date`: running prediction

# go to script dir
cd /home/feiw/feiw/frag-gnn
# load env
source ./FRAG-GNN-GPU2/bin/activate
# nist20_d4_wr00_inchikey_test_dags
srun python ./scripts/ms2c/run_fraggn_ms2c.py --batch_size=4 \
--frag_dp=${SLURM_TMPDIR}/nist20_d4_wr00_inchikey_test_dags \
--task_dp="data/ms2c/pubchem/pred_inputs/pubchem_nist20_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50" \
--custom_fp="config/nist20_d4_wr00_inchikey/fraggnn_d4_nodemlp_mnce_nocp.yml" \
--ckpt_fp="./saved_ckpts/kme8fry7_fraggnn_d4_nodemlp_mnce_nocp/ckpt/model-epoch=098.ckpt" \
--device="cpu"

# nist20_d4_wr00_scaffold_test_dags
srun python ./scripts/ms2c/run_fraggn_ms2c.py --batch_size=4 \
--frag_dp=${SLURM_TMPDIR}/nist20_d4_wr00_scaffold_test_dags \
--task_dp="data/ms2c/pubchem/pred_inputs/pubchem_nist20_d4_wr00_scaffold_test_10ppm_MORGAN-R2_50" \
--custom_fp="config/nist20_d4_wr00_scaffold/fraggnn_d4_nodemlp_mnce_nocp.yml" \
--ckpt_fp="./saved_ckpts/gmpfciq9_fraggnn_d4_nodemlp_mnce_nocp/ckpt/model-epoch=096.ckpt" \
--device="cpu"

# record start
echo `date`: Job ${SLURM_JOB_ID} is done