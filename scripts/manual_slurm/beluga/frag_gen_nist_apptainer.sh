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


DATASET_NAME=nist20mona23v3
MAX_DEPTH=3
NB_FLAG=False
SAVE_NAME=nist20mona23v3_d3_h4_ma_mi
DSETS='nist20 mona23'
PREC='M+H+ M-H-'
INST='QTOF FT'

echo DATASET_NAME: ${DATASET_NAME}
echo MAX_DEPTH: ${MAX_DEPTH}
echo SAVE_NAME: ${SAVE_NAME}
echo NB_FLAG: ${NB_FLAG}

# create local dir
echo create local proc_dp: ${SLURM_TMPDIR}/proc
mkdir -p ${SLURM_TMPDIR}/proc
echo create local frag_dp: ${SLURM_TMPDIR}/frag/${DATASET_NAME}
mkdir -p ${SLURM_TMPDIR}/frag/${DATASET_NAME}

# copy file to slrum local dir
echo copy from ./data/proc/${DATASET_NAME} to ${SLURM_TMPDIR}/data/proc

# copy data
rsync -av ./data/proc/${DATASET_NAME} ${SLURM_TMPDIR}/proc 
echo list file in local proc dir: $(ls ${SLURM_TMPDIR}/proc/${DATASET_NAME})

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
# run dag gen
echo running dag gen

apptainer exec -C -B ${SLURM_TMPDIR}:$HOME -W ${SLURM_TMPDIR} ${SLURM_TMPDIR}/fragnnet_cpu.sif \
    ${SLURM_TMPDIR}/fragnnet_cpu.sif \
    python preproc_scripts/03_prepare_dag_feats.py \
        --frag_dp=${SLURM_TMPDIR}/frag/${SAVE_NAME} \
        --proc_dp=./data/proc/${DATASET_NAME}  \
        --max_depth=${MAX_DEPTH} \
        --wandb_mode=off \
        --wandb_run_name=fool \
        --prec_types ${PREC}\
        --inst_types ${INST} \
        --dsets ${DSETS} \
        --nb_isomorphic ${NB_FLAG}

# go to frag directory
cd ${SLURM_TMPDIR}/frag/

tar -cf ${SAVE_NAME}.tar ${SAVE_NAME}
DEST_DP=/scratch/feiw/frag-gnn-data/frags/d${MAX_DEPTH}_h4_isoFalse
echo create dir at ${DEST_DP}
mkdir -p ${DEST_DP}
echo copy tar: ${SAVE_NAME}.tar to ${DEST_DP}
cp ${SAVE_NAME}.tar ${DEST_DP}

# record ending
echo `date`: Job ${SLURM_JOB_ID} finished running
