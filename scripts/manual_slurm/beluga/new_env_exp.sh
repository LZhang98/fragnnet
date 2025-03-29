
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
rsync -av --exclude=".*" --exclude="__pycache__" --exclude="*.so" src ${SLURM_TMPDIR}
cp setup.cfg ${SLURM_TMPDIR}/
cp setup.py ${SLURM_TMPDIR}/

# instal fragnnet
pip install -I .

# Copy files
# create local dir
DATASET_NAME=nist23mona23
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
cd ${FRAGNNET_DP}
rsync -av ./data/proc/${DATASET_NAME} ${SLURM_TMPDIR}/data/proc 
echo list file in local proc dir: $(ls ${SLURM_TMPDIR}/data/proc/${DATASET_NAME})

# copy proc file
SPLIT=nist20_mona23_d4_wr00_inchikey
rsync -av ./data/split/${SPLIT} ${SLURM_TMPDIR}/data/split 
echo list file in local proc dir: $(ls ${SLURM_TMPDIR}/data/split/${SPLIT})

# copy config files
cp ${FRAGNNET_DP}/config/template.yml  ${SLURM_TMPDIR}/config/template.yml  

# copy frag file
cp ${FRAGNNET_DP}/data/frags/nist23mona23_ma_mi_d4_h4_isoFalse.tar  ${SLURM_TMPDIR}/data/frag/
cd ${SLURM_TMPDIR}/data/frag
tar xf nist23mona23_ma_mi_d4_h4_isoFalse.tar

#echo list file in local proc dir: $(ls ${SLURM_TMPDIR}/proc/${DATASET_NAME})
cd ${SLURM_TMPDIR}
python ./preproc_scripts/04_prepare_split.py \
    --split_type random \
    --split_key inchikey_s \
    --frag_dp ./data/frag/nist23mona23_ma_mi_d4_h4_isoFalse \
    --proc_dp ./data/proc/nist23mona23 \
    --split_dp ./data/split/nist23_ma_mi \
    --inst_types FT QTOF \
    --prec_types [M+H]+ [M-H]- \
    --primary_dsets nist23_hr1 nist23_hr2 \
    --ion_mode P N \
    --frag_mode CID HCD

# soft link back to home directy for wandb
ln -s ${FRAGNNET_DP}/wandb wandb
