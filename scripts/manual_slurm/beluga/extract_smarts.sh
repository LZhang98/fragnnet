
#!/bin/bash
#SBATCH --account=rrg-dwishart
#SBATCH --nodes=1
#SBATCH --job-name=extract_smarts
#SBATCH --mem=90G
#SBATCH --cpus-per-task=40
#SBATCH --time=0-1:00
#SBATCH --mail-user=fw4@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --output=/home/feiw/feiw/frag-gnn/jobs/extract_smarts_%j.out
#SBATCH --error=/home/feiw/feiw/frag-gnn/jobs/extract_smarts_%j.err
#SBATCH --open-mode=append

FRAGNNET_DP=/home/feiw/feiw/frag-gnn

# load modules
module load StdEnv/2023
module load python/3.11
module load rdkit/2023.09.5

# go to slurm and create virtual env
cd  ${SLURM_TMPDIR}

virtualenv FRAG-GNN
source FRAG-GNN/bin/activate
pip install --no-index --upgrade pip

pip install --no-index torch==2.4.1
pip install --no-index torch_geometric==2.6.1 
pip install --no-index torch_scatter
pip install --no-index pandas
pip install --no-index wandb
pip install --no-index tqdm
pip install --no-index scipy
pip install --no-index joblib==1.3.2
pip install --no-index matplotlib
pip install --no-index pytorch_lightning==2.4.0
pip install --no-index dgl==2.4.0
pip install --no-index omegaconf
pip install --no-index cython==3.0.12

# intall from whl
cp ${FRAGNNET_DP}/wheels/pyteomics-4.7.5-py3-none-any.whl ${SLURM_TMPDIR}/pyteomics-4.7.5-py3-none-any.whl 
pip install pyteomics-4.7.5-py3-none-any.whl 

# go to source location
cd ${FRAGNNET_DP}
# copy source code
rsync -av --exclude=".*" --exclude="__pycache__" preproc_scripts ${SLURM_TMPDIR}
rsync -av --exclude=".*" --exclude="__pycache__" scripts ${SLURM_TMPDIR}
#rsync -av --exclude=".*" --exclude="__pycache__" --exclude="*.so" src ${SLURM_TMPDIR}
#cp setup.cfg ${SLURM_TMPDIR}/
#cp setup.py ${SLURM_TMPDIR}/

# instal fragnnet
pip install -I .

# Copy files
# create local dir
DATASET_NAME=nist20mona23v3
FRAG_TAR=nist20mona23v3_ma_mi_d3_h4_isoFalse.tar
SPLIT=nist20v3_ma_mi_inchikey
CONFIG=config/exp_fraggnn_ma_mi/fraggnn_d3_ma_mi.yml
OUTPUT_DP=data/smarts/fraggnn_d3_ma_mi

echo create local proc_dp: ${SLURM_TMPDIR}/data/proc
mkdir -p ${SLURM_TMPDIR}/data/proc
echo create local frag_dp: ${SLURM_TMPDIR}/data/frag/
mkdir -p ${SLURM_TMPDIR}/data/frag/
echo create local split_dp: ${SLURM_TMPDIR}/data/split/
mkdir -p ${SLURM_TMPDIR}/data/split/
# create local config dir
echo create local config_dp: ${SLURM_TMPDIR}/config/
mkdir -p ${SLURM_TMPDIR}/config/

# copy proc file
rsync -av ${FRAGNNET_DP}/data/proc/${DATASET_NAME} ${SLURM_TMPDIR}/data/proc 
echo list file in local proc dir: $(ls ${SLURM_TMPDIR}/data/proc/${DATASET_NAME})

# copy split file
rsync -av ${FRAGNNET_DP}/data/split/${SPLIT} ${SLURM_TMPDIR}/data/split 
echo list file in local proc dir: $(ls ${SLURM_TMPDIR}/data/split/${SPLIT})

# copy config files
cp ${FRAGNNET_DP}/config/template.yml  ${SLURM_TMPDIR}/config/template.yml  
rsync --relative ${FRAGNNET_DP}/${CONFIG} ${SLURM_TMPDIR}

# copy frag file
cp ${FRAGNNET_DP}/data/frag/${FRAG_TAR}  ${SLURM_TMPDIR}/data/frag/
cd ${SLURM_TMPDIR}/data/frag
tar xf ${FRAG_TAR}

# go to slurm temp dir
cd ${SLURM_TMPDIR}
# soft link back to home directy for wandb
ln -s ${FRAGNNET_DP}/wandb wandb

python ./preproc_scripts/no_dag_exp/01_create_smarts.py \
	--output_dp=${OUTPUT_DP} \
	--custom_fp=${CONFIG} \
	--batch_size=320 \
	--num_workers=8 \
	--n_jobs=40
