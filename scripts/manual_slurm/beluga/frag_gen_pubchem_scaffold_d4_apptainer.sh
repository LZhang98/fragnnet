#!/bin/bash
#SBATCH --account=rrg-dwishart
#SBATCH --nodes=1
#SBATCH --job-name=frag_gen_pubchem_scaffold_d4_apptainer
#SBATCH --mem=90G
#SBATCH --cpus-per-task=40
#SBATCH --time=0-2:00
#SBATCH --mail-user=fw4@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --output=/home/feiw/feiw/frag-gnn/jobs/frag_gen_pubchem_scaffold_d4_apptainer%j.out
#SBATCH --error=/home/feiw/feiw/frag-gnn/jobs/frag_gen_pubchem_scaffold_d4_apptainer%j.err
#SBATCH --open-mode=append

# record start
echo `date`: Job ${SLURM_JOB_ID} is allocated resources

#load modules
module purge
module load StdEnv/2023
module load python/3.10

# go to script dir
cd /home/feiw/feiw/frag-gnn

MAX_DEPTH=4
RUN_NAME=run_250218
DATASET_NAME=pubchem_nist20mona23v3_d4_wr00_scaffold_test_10ppm_MORGAN-R2_50
PROC_DP=/scratch/feiw/frag-gnn-data/ms2c/pubchem/${RUN_NAME}/proc/${DATASET_NAME}
DEST_DP=/scratch/feiw/frag-gnn-data/ms2c/pubchem/${RUN_NAME}/frags/d${MAX_DEPTH}_h4_isoFalse

echo RUN_NAME: ${RUN_NAME}
echo MAX_DEPTH: ${MAX_DEPTH}
echo PROC_DP: ${PROC_DP}
echo DEST_DP: ${DEST_DP}

# create local dir
echo create local proc_dp: ${SLURM_TMPDIR}/proc
mkdir -p ${SLURM_TMPDIR}/proc
echo create local frag_dp: ${SLURM_TMPDIR}/frag/${DATASET_NAME}
mkdir -p ${SLURM_TMPDIR}/frag/${DATASET_NAME}

# copy file to slrum local dir
rsync -av ${PROC_DP} ${SLURM_TMPDIR}/proc 
echo list file in local proc dir: $(ls ${SLURM_TMPDIR}/proc/${DATASET_NAME})

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
echo install frag-gnn
apptainer exec -C -B ${SLURM_TMPDIR}:$HOME -W ${SLURM_TMPDIR} ${SLURM_TMPDIR}/fragnnet_cpu.sif  pip install -I . --user

# run dag gen
echo running dag gen

apptainer exec -C -B ${SLURM_TMPDIR}:$HOME -W ${SLURM_TMPDIR} ${SLURM_TMPDIR}/fragnnet_cpu.sif \
    python preproc_scripts/pubchem_ms2c/04_run_ms2c_frag_gen.py \
    --frag_dp=frag/${DATASET_NAME} \
    --proc_dp=proc/${DATASET_NAME} \
    --max_depth=${MAX_DEPTH}

# go to frag directory
cd ${SLURM_TMPDIR}/frag/
tar -cf ${DATASET_NAME}.tar ${DATASET_NAME}

echo create dir at ${DEST_DP}
mkdir -p ${DEST_DP}
echo copy tar: ${DATASET_NAME}.tar to ${DEST_DP}
cp ${DATASET_NAME}.tar ${DEST_DP}

# record ending
echo `date`: Job ${SLURM_JOB_ID} finished running