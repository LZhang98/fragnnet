#!/bin/bash
#SBATCH --account=rrg-dwishart
#SBATCH --nodes=1
#SBATCH --job-name=iceberg_gen_pubchem_inchikey
#SBATCH --mem=90G
#SBATCH --cpus-per-task=40
#SBATCH --time=0-4:00
#SBATCH --mail-user=fw4@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --output=/home/feiw/feiw/frag-gnn/jobs/iceberg_gen_pubchem_inchikey_%j.out
#SBATCH --error=/home/feiw/feiw/frag-gnn/jobs/iceberg_gen_pubchem_inchikey_%j.err
#SBATCH --open-mode=append

# record start
echo `date`: Job ${SLURM_JOB_ID} is allocated resources

# record start
echo `date`: Job ${SLURM_JOB_ID} is allocated resources

#load modules
module load StdEnv/2023
module load python/3.10

# go to script dir
cd /home/feiw/feiw/frag-gnn

SAVE_FP=./nist20v3_inchikey_iceberg_inten_100_s0.pkl
MAGMA_DP=./pred_magma_nist20v3_inchikey_iceberg_gen_s0_i7k9gc3d
ICEBERG_GEN_CKPT_FP=./saved_ckpts/nist20v3_inchikey_iceberg_gen_s0_i7k9gc3d.ckpt
ICEBERG_INTERN_CKPT_FP=./saved_ckpts/nist20v3_inchikey_iceberg_inten_100_s0_c8iwu7il.ckpt
PROC_DP=./data/ms2c/pubchem/proc/pubchem_nist20mona23v3_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50

# create local dir
echo create local proc_dp: ${SLURM_TMPDIR}/proc
mkdir -p ${SLURM_TMPDIR}/proc
#echo create local frag_dp: ${SLURM_TMPDIR}/frag/${DATASET_NAME}
#mkdir -p ${SLURM_TMPDIR}/frag/${DATASET_NAME}

# copy file to slrum local dir
rsync -av ${PROC_DP}/ ${SLURM_TMPDIR}/proc 
echo list file in local proc dir: $(ls ${SLURM_TMPDIR}/proc/)
cp ${ICEBERG_GEN_CKPT_FP} ${SLURM_TMPDIR}/iceberg_gen.ckpt
cp ${ICEBERG_INTERN_CKPT_FP} ${SLURM_TMPDIR}/iceberg_inten.ckpt

# copy apptainer container
module load apptainer/1.2.4
cp ./containers/fragnnet_cpu.sif ${SLURM_TMPDIR}/fragnnet_cpu.sif

# copy source code
rsync -av --exclude=".*" --exclude="__pycache__" preproc_scripts ${SLURM_TMPDIR}
rsync -av --exclude=".*" --exclude="__pycache__" scripts ${SLURM_TMPDIR}
rsync -av --exclude=".*" --exclude="__pycache__"  --exclude="*.so" src ${SLURM_TMPDIR}
cp setup.cfg ${SLURM_TMPDIR}/
cp setup.py ${SLURM_TMPDIR}/

# install frag-gnn
apptainer exec -C -B ${SLURM_TMPDIR}:$HOME -W ${SLURM_TMPDIR} ${SLURM_TMPDIR}/fragnnet_cpu.sif  pip install -I . --user

# run dag gen
apptainer exec -C -B ${SLURM_TMPDIR}:$HOME -W ${SLURM_TMPDIR} ${SLURM_TMPDIR}/fragnnet_cpu.sif \
    python scripts/ms2c/pubchem_v1/run_iceberg_ms2c_predict_val.py \
        --device=cpu \
        --save_fp=./nist20v3_inchikey_iceberg_inten_100_s0.pkl \
        --magma_dp=./magma_dp \
        --iceberg_gen_ckpt_fp=iceberg_gen.ckpt \
        --iceberg_intern_ckpt_fp=iceberg_gen.ckpt \
        --proc_dp=./proc

cd ${SLURM_TMPDIR}
tar -cf pred_magma_nist20v3_inchikey_iceberg_gen_s0_i7k9gc3d.tar pred_magma_nist20v3_inchikey_iceberg_gen_s0_i7k9gc3d
mkdir -p /scratch/feiw/frag-gnn-data/ms2c/pubchem/magma/
cp pred_magma_nist20v3_inchikey_iceberg_gen_s0_i7k9gc3d.tar /scratch/feiw/frag-gnn-data/ms2c/pubchem/magma/

# record ending
echo `date`: Job ${SLURM_JOB_ID} finished running


