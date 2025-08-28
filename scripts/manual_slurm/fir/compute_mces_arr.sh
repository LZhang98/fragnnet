
#!/bin/bash
#SBATCH --account=rrg-dwishart
#SBATCH --nodes=1
#SBATCH --job-name=compute_mces
#SBATCH --mem=192G
#SBATCH --cpus-per-task=192
#SBATCH --mail-user=fw4@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --output=./jobs/compute_mces_array_job_%A_%a.out
#SBATCH --error=./jobs/compute_mces_array_job_%A_%a.err
#SBATCH --array=1-106
#SBATCH --time=02:00:00

# load modules
module purge
module load StdEnv/2023
module load python/3.11
module load rdkit/2023.09.5

# load CPLEX
CPLEX_HOME=/home/feiw/ibm/ILOG/CPLEX_Studio2211/cplex/
PATH=$CPLEX_HOME/bin/x86-64_linux/:$PATH
INPUT_FILE=./nist_23_inputs/mces_input_${SLURM_ARRAY_TASK_ID}.h5
MYOPIC_SRC=./wheels/myopic_mces-1.0.1.tar.gz

# record start
echo `date`: Job ${SLURM_JOB_ID} is allocated resources
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"

# create local env
virtualenv ${SLURM_TMPDIR}/MCES
source ${SLURM_TMPDIR}/MCES/bin/activate
pip install --no-index --upgrade pip
pip install --no-index networkx==3.0 scipy==1.10.1 pulp==2.7.0 joblib==1.2.0 h5py
python -m pip install --no-deps ${MYOPIC_SRC}

# run python script
#cd ${SLURM_TMPDIR}
python -m myopic_mces.myopic_mces \
    --threshold 10 --solver CPLEX_CMD \
    --solver_onethreaded --solver_no_msg \
    --hdf5_mode $INPUT_FILE

# record ending
echo `date`: Job ${SLURM_JOB_ID} finished running