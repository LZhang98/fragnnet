#!/bin/bash
#SBATCH --account=rrg-dwishart
#SBATCH --nodes=1
#SBATCH --job-name=iceberg_gen_pubchem_inchikey
#SBATCH --mem=90G
#SBATCH --cpus-per-task=40
#SBATCH --time=0-2:00
#SBATCH --mail-user=fw4@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --output=/home/feiw/feiw/frag-gnn/jobs/iceberg_gen_pubchem_inchikey_%j.out
#SBATCH --error=/home/feiw/feiw/frag-gnn/jobs/iceberg_gen_pubchem_inchikey_%j.err
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

python scripts/ms2c/pubchem_v1/run_iceberg_ms2c_predict_val.py \
	--device=cpu\
    --save_fp=${SLURM_TMPDIR}/nist20v3_inchikey_iceberg_inten_100_s0.pkl\
    --magma_dp=${SLURM_TMPDIR}/pred_magma_nist20v3_inchikey_iceberg_gen_s0_i7k9gc3d\
    --iceberg_gen_ckpt_fp=./saved_ckpts/nist20v3_inchikey_iceberg_gen_s0_i7k9gc3d/nist20v3_inchikey_iceberg_gen_s0_i7k9gc3d.ckpt\
    --iceberg_intern_ckpt_fp=./saved_ckpts/nist20v3_inchikey_iceberg_inten_100_s0_c8iwu7il/nist20v3_inchikey_iceberg_inten_100_s0_c8iwu7il.ckpt\
    --proc_dp=./data/ms2c/pubchem/proc/pubchem_nist20mona23v3_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50

cd ${SLURM_TMPDIR}
tar -cf pred_magma_nist20v3_inchikey_iceberg_gen_s0_i7k9gc3d.tar pred_magma_nist20v3_inchikey_iceberg_gen_s0_i7k9gc3d
mkdir -p /scratch/feiw/frag-gnn-data/ms2c/pubchem/magma/
cp pred_magma_nist20v3_inchikey_iceberg_gen_s0_i7k9gc3d.tar /scratch/feiw/frag-gnn-data/ms2c/pubchem/magma/

# record ending
echo `date`: Job ${SLURM_JOB_ID} finished running


