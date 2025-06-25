
#!/bin/bash
#SBATCH --account=rrg-dwishart
#SBATCH --nodes=1
#SBATCH --job-name=compute_mces
#SBATCH --mem=90G
#SBATCH --cpus-per-task=40
#SBATCH --time=0-1:00
#SBATCH --mail-user=fw4@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --output=/home/feiw/feiw/frag-gnn/jobs/compute_mces_%j.out
#SBATCH --error=/home/feiw/feiw/frag-gnn/jobs/compute_mces_%j.err
#SBATCH --open-mode=append

# load CPLEX
CPLEX_HOME=/home/feiw/ibm/ILOG/CPLEX_Studio2211/cplex/
PATH=$CPLEX_HOME/bin/x86-64_linux/:$PATH
INPUT_FILE=/home/feiw/feiw/frag-gnn/data/mces-data/biostrcutures/biostructures_sampled_1.hdf5

# record start
echo `date`: Job ${SLURM_JOB_ID} is allocated resources
#load modules
module load StdEnv/2023
module load python/3.11
module load rdkit/2022.09.5

# create local env
cd ${SLURM_TMPDIR}
virtualenv MCES
source MCES/bin/activate
pip install --no-index --upgrade pip

pip install --no-index networkx==3.0
pip install --no-index pulp==2.7.0
pip install --no-index scipy==1.10.1
pip install --no-index joblib==1.2.0
pip install --no-index h5py

# run python script
cd ${SLURM_TMPDIR}
python -m myopic_mces.myopic_mces \
    --threshold 10 --solver CPLEX_CMD \
    --solver_onethreaded --solver_no_msg
    --hdf5_mode $INPUT_FILE \ 
    mce_output

# record ending
echo `date`: Job ${SLURM_JOB_ID} finished running