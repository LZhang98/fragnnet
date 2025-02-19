#!/bin/bash
#SBATCH --account=rrg-dwishart
#SBATCH --nodes=1
#SBATCH --job-name=iceberg_pubchem_inchikey
#SBATCH --mem=90G
#SBATCH --cpus-per-task=40
#SBATCH --time=0-24:00
#SBATCH --mail-user=fw4@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --output=/home/feiw/feiw/frag-gnn/jobs/iceberg_pubchem_inchikey_%A_%a.out
#SBATCH --error=/home/feiw/feiw/frag-gnn/jobs/iceberg_pubchem_inchikey_%A_%a.err
#SBATCH --open-mode=append
#SBATCH --array=0-4

# record start
echo `date`: Job ${SLURM_JOB_ID} is allocated resources
echo SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}

#load modules
module load StdEnv/2023
module load python/3.10

# go to script dir
cd /home/feiw/feiw/frag-gnn

RUN_NAME=run_250218
ICEBERG_GEN_RUN_IDS=('i7k9gc3d' 'mz3s3h62' '6f4nes2g' 'lp60e1nq' '3zjhzrss')
ICEBERG_INETRN_RUN_IDS=('c8iwu7il' 'rtt8i90g' 'acgyna2a' '683gijym' 'yeatgz7o')
DATASET_NAME=pubchem_nist20mona23v3_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50

task_id=${SLURM_ARRAY_TASK_ID}
iceberg_gen_run_id=${ICEBERG_GEN_RUN_IDS[$task_id]}
iceberg_inten_run_id=${ICEBERG_INETRN_RUN_IDS[$task_id]}

echo task_id: ${task_id}
echo iceberg_gen_run_id: ${iceberg_gen_run_id}
echo iceberg_inten_run_id: ${iceberg_inten_run_id}

SAVE_FP=nist20v3_inchikey_iceberg_inten_100_s${task_id}.pkl
MAGMA_DP=pred_magma_nist20v3_inchikey_iceberg_gen_s${task_id}_${iceberg_gen_run_id}
ICEBERG_GEN_CKPT_FP=./saved_ckpts/nist20v3_inchikey_iceberg_gen_s${task_id}_${iceberg_gen_run_id}.ckpt
ICEBERG_INTERN_CKPT_FP=./saved_ckpts/nist20v3_inchikey_iceberg_inten_100_s${task_id}_${iceberg_inten_run_id}.ckpt
PROC_DP=/scratch/feiw/frag-gnn-data/ms2c/pubchem/${RUN_NAME}/proc/${DATASET_NAME}

MAGMA_SAVE_DP=/scratch/feiw/frag-gnn-data/ms2c/pubchem/${RUN_NAME}/magma
PREDICTED_SAVE_DP=/scratch/feiw/frag-gnn-data/ms2c/pubchem/${RUN_NAME}/predicted

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

# run iceberg gen
apptainer exec -C -B /dev:/dev -B ${SLURM_TMPDIR}:$HOME -W ${SLURM_TMPDIR} ${SLURM_TMPDIR}/fragnnet_cpu.sif \
    python scripts/ms2c/pubchem_v1/run_iceberg_ms2c_predict_val.py \
        --iceberg_gen_device=cpu \
        --save_fp=${SAVE_FP} \
        --magma_dp=${MAGMA_DP} \
        --iceberg_gen_ckpt_fp=iceberg_gen.ckpt \
        --iceberg_intern_ckpt_fp=iceberg_inten.ckpt \
        --proc_dp=./proc \
        --run_iceberg_gen=True \
        --run_iceberg_intern=False

cd ${SLURM_TMPDIR}
echo create predicted save dir: ${MAGMA_SAVE_DP}
mkdir -p ${MAGMA_SAVE_DP}
tar -czf ${MAGMA_DP}.tar.gz ${MAGMA_DP}
echo copy ${MAGMA_DP}.tar.gz to ${MAGMA_SAVE_DP}
cp ${MAGMA_DP}.tar.gz ${MAGMA_SAVE_DP}

apptainer exec -C -B /dev:/dev -B ${SLURM_TMPDIR}:$HOME -W ${SLURM_TMPDIR} ${SLURM_TMPDIR}/fragnnet_cpu.sif \
    python scripts/ms2c/pubchem_v1/run_iceberg_ms2c_predict_val.py \
        --iceberg_intern_device=cpu \
        --save_fp=${SAVE_FP} \
        --magma_dp=${MAGMA_DP} \
        --iceberg_gen_ckpt_fp=iceberg_gen.ckpt \
        --iceberg_intern_ckpt_fp=iceberg_inten.ckpt \
        --proc_dp=./proc \
        --run_iceberg_gen=False \
        --run_iceberg_chunk_size=10000 \
        --run_iceberg_intern=True

cd ${SLURM_TMPDIR}
echo create predicted save dir: ${PREDICTED_SAVE_DP}
mkdir -p ${PREDICTED_SAVE_DP}
gzip ${SAVE_FP} 
echo copy ${SAVE_FP}.gz to ${PREDICTED_SAVE_DP}/${SAVE_FP}.gz
cp ${SAVE_FP}.gz ${PREDICTED_SAVE_DP}/${SAVE_FP}.gz

# record ending
echo `date`: Job ${SLURM_JOB_ID} finished running