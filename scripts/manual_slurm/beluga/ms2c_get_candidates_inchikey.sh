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

#load modules
module load StdEnv/2023
module load python/3.10
moudle load rdkit/2023.09.5

# go to script dir
cd /home/feiw/feiw/frag-gnn

SPLIT_DP=data/split/nist20mona23v3_d4_wr00_inchikey
PROC_DP=./data/proc/nist20mona23v3
RUN_NAME=run_20241127

# create local dir
echo create local proc_dp: ${SLURM_TMPDIR}/proc
mkdir -p ${SLURM_TMPDIR}/proc

# copy file to slrum local dir
rsync -av ${PROC_DP}/ ${SLURM_TMPDIR}/proc 
echo list file in local proc dir: $(ls ${SLURM_TMPDIR}/proc/)
rysnc -av ${SPLIT_DP}/ ${SLURM_TMPDIR}/split 
echo list file in local split dir: $(ls ${SLURM_TMPDIR}/split/)

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
        --split_dp=split \
        --proc_dp=proc \
        --split_type=test \
        --candidate_dp=candidates

# run 03_perpare_ms2c_mol_and_spec_df
apptainer exec -C -B /dev:/dev -B ${SLURM_TMPDIR}:$HOME -W ${SLURM_TMPDIR} ${SLURM_TMPDIR}/fragnnet_cpu.sif \
    python preproc_scripts/pubchem_ms2c/03_perpare_ms2c_mol_and_spec_df.py \
        --candidate_pkl_fp=candidates/pubchem_nist20mona23v3_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50.pkl.gz \
        --proc_dp=proc \
        --split_dp=split \
        --split_type=test \
        --ms2c_data_dp=ms2c_data_dp

cd ${SLURM_TMPDIR}
mkdir -p /scratch/feiw/frag-gnn-data/ms2c/pubchem/${RUN_NAME}/candidates
rsync -av  candidates /scratch/feiw/frag-gnn-data/ms2c/pubchem/${RUN_NAME}/candidates

mkdir -p /scratch/feiw/frag-gnn-data/ms2c/pubchem/${RUN_NAME}/proc
rsync -av  proc /scratch/feiw/frag-gnn-data/ms2c/pubchem/${RUN_NAME}/proc

# record ending
echo `date`: Job ${SLURM_JOB_ID} finished running