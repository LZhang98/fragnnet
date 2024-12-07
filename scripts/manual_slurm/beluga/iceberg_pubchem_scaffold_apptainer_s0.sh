#!/bin/bash
#SBATCH --account=rrg-dwishart
#SBATCH --nodes=1
#SBATCH --job-name=iceberg_gen_pubchem_scaffold
#SBATCH --mem=90G
#SBATCH --cpus-per-task=40
#SBATCH --time=0-24:00
#SBATCH --mail-user=fw4@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --output=/home/feiw/feiw/frag-gnn/jobs/iceberg_gen_pubchem_scaffold_%j.out
#SBATCH --error=/home/feiw/feiw/frag-gnn/jobs/iceberg_gen_pubchem_scaffold_%j.err
#SBATCH --open-mode=append

# record start
echo `date`: Job ${SLURM_JOB_ID} is allocated resources

#load modules
module load StdEnv/2023
module load python/3.10
module load rdkit/2023.09.5

# go to script dir
cd /home/feiw/feiw/frag-gnn

SAVE_FP=nist20v3_scaffold_iceberg_inten_100_s0.pkl
MAGMA_DP=pred_magma_nist20v3_scaffold_iceberg_gen_s0_4pweeg45
ICEBERG_GEN_CKPT_FP=./saved_ckpts/nist20v3_scaffold_iceberg_gen_s0_4pweeg45.ckpt
ICEBERG_INTERN_CKPT_FP=./saved_ckpts/nist20v3_scaffold_iceberg_inten_100_s0_riyr41po.ckpt
PROC_DP=./data/ms2c/pubchem/proc/pubchem_nist20mona23v3_d4_wr00_scaffold_test_10ppm_MORGAN-R2_50

# create local dir
echo create local proc_dp: ${SLURM_TMPDIR}/proc
mkdir -p ${SLURM_TMPDIR}/proc

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
rsync -av --exclude=".*" --exclude="__pycache__" --exclude="*.so" src ${SLURM_TMPDIR}
cp setup.cfg ${SLURM_TMPDIR}/
cp setup.py ${SLURM_TMPDIR}/

# install frag-gnn
apptainer exec -C -B ${SLURM_TMPDIR}:$HOME -W ${SLURM_TMPDIR} ${SLURM_TMPDIR}/fragnnet_cpu.sif  pip install -I . --user

# run dag gen
apptainer exec -C -B /dev:/dev -B ${SLURM_TMPDIR}:$HOME -W ${SLURM_TMPDIR} ${SLURM_TMPDIR}/fragnnet_cpu.sif \
    python scripts/ms2c/pubchem_v1/run_iceberg_ms2c_predict_val.py \
        --iceberg_gen_device=cpu \
        --save_fp=${SAVE_FP} \
        --magma_dp=${MAGMA_DP} \
        --iceberg_gen_ckpt_fp=iceberg_gen.ckpt \
        --iceberg_intern_ckpt_fp=iceberg_inten.ckpt \
        --proc_dp=./proc \
        --run_iceberg_gen=True \
        --iceberg_gen_use_cached=False \
        --run_iceberg_chunk_size=10000 \
        --run_iceberg_intern=False

cd ${SLURM_TMPDIR}
mkdir -p /scratch/feiw/frag-gnn-data/ms2c/pubchem/magma/
tar -czf ${MAGMA_DP}.tar.gz ${MAGMA_DP}
cp ${MAGMA_DP}.tar.gz /scratch/feiw/frag-gnn-data/ms2c/pubchem/magma/

apptainer exec -C -B /dev:/dev -B ${SLURM_TMPDIR}:$HOME -W ${SLURM_TMPDIR} ${SLURM_TMPDIR}/fragnnet_cpu.sif \
    python scripts/ms2c/pubchem_v1/run_iceberg_ms2c_predict_val.py \
        --iceberg_intern_device=cpu \
        --save_fp=${SAVE_FP} \
        --magma_dp=${MAGMA_DP} \
        --iceberg_gen_ckpt_fp=iceberg_gen.ckpt \
        --iceberg_intern_ckpt_fp=iceberg_inten.ckpt \
        --proc_dp=./proc \
        --run_iceberg_gen=False \
        --run_iceberg_intern=True

cd ${SLURM_TMPDIR}       
mkdir -p /scratch/feiw/frag-gnn-data/ms2c/pubchem/predicted/
cp ${SAVE_FP} /scratch/feiw/frag-gnn-data/ms2c/pubchem/predicted/${SAVE_FP}

# record ending
echo `date`: Job ${SLURM_JOB_ID} finished running


