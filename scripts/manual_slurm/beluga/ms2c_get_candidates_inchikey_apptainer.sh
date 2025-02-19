#!/bin/bash
#SBATCH --account=rrg-dwishart
#SBATCH --nodes=1
#SBATCH --job-name=ms2c_get_candidates_inchikey_apptainer
#SBATCH --mem=90G
#SBATCH --cpus-per-task=40
#SBATCH --time=0-4:00
#SBATCH --mail-user=fw4@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --output=/home/feiw/feiw/frag-gnn/jobs/ms2c_get_candidates_inchikey_apptainer_%j.out
#SBATCH --error=/home/feiw/feiw/frag-gnn/jobs/ms2c_get_candidates_inchikey_apptainer_%j.err
#SBATCH --open-mode=append

# record start
echo `date`: Job ${SLURM_JOB_ID} is allocated resources

#load modules
module purge
module load StdEnv/2023

# go to script dir
cd /home/feiw/feiw/frag-gnn

DATASET_NAME=nist20mona23v3_d4_wr00_inchikey
SPLIT_DP=data/split/${DATASET_NAME}
PROC_DP=./data/proc/nist20mona23v3
DB_FP=./data/ms2c/pubchem/pubchem.sqlite

RUN_NAME=run_250218
CANDIDATE_DP=/scratch/feiw/frag-gnn-data/ms2c/pubchem/${RUN_NAME}/candidates
CANDIDATE_PROC_DP=/scratch/feiw/frag-gnn-data/ms2c/pubchem/${RUN_NAME}/proc

# create local dir
echo create local proc_dp: ${SLURM_TMPDIR}/proc
mkdir -p ${SLURM_TMPDIR}/proc

# copy file to slrum local dir
rsync -av --mkpath ${PROC_DP}/ ${SLURM_TMPDIR}/proc 
echo list file in local proc dir: $(ls ${SLURM_TMPDIR}/proc/)
rsync -av --mkpath ${SPLIT_DP}/ ${SLURM_TMPDIR}/split/${DATASET_NAME} 
echo list file in local split dir: $(ls ${SLURM_TMPDIR}/split/${DATASET_NAME} )
rsync -av --mkpath ${DB_FP} ${SLURM_TMPDIR}/pubchem.sqlite

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

# run 02_prepare_ms2c_candidates
apptainer exec -C -B /dev:/dev -B ${SLURM_TMPDIR}:$HOME -W ${SLURM_TMPDIR} ${SLURM_TMPDIR}/fragnnet_cpu.sif \
    python preproc_scripts/pubchem_ms2c/02_prepare_ms2c_candidates.py  \
        --split_dp=split/${DATASET_NAME} \
        --proc_dp=proc \
        --split_type=test \
        --db_file=./pubchem.sqlite \
        --candidate_dp=candidates

CANDIDATE_FILENAME=$(ls -1 ${SLURM_TMPDIR}/candidates| head -n 1)
CANDIDATE_FILENAME_NO_EXT="${CANDIDATE_FILENAME%%.*}"

# run 03_perpare_ms2c_mol_and_spec_df
apptainer exec -C -B /dev:/dev -B ${SLURM_TMPDIR}:$HOME -W ${SLURM_TMPDIR} ${SLURM_TMPDIR}/fragnnet_cpu.sif \
    python preproc_scripts/pubchem_ms2c/03_perpare_ms2c_mol_and_spec_df.py \
        --candidate_pkl_fp=candidates/${CANDIDATE_FILENAME} \
        --proc_dp=proc \
        --split_dp=split/${DATASET_NAME} \
        --split_type=test \
        --ms2c_data_dp=ms2c_data_dp

cd ${SLURM_TMPDIR}
echo create candidate save dir: ${CANDIDATE_DP}
mkdir -p ${CANDIDATE_DP}
rsync -av candidates/ ${CANDIDATE_DP}

echo create proc save dir: ${CANDIDATE_PROC_DP}
mkdir -p ${CANDIDATE_PROC_DP}
rsync -av --mkpath ms2c_data_dp/ ${CANDIDATE_PROC_DP}

# record ending
echo `date`: Job ${SLURM_JOB_ID} finished running