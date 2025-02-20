#!/bin/bash
#SBATCH --account=rrg-dwishart
#SBATCH --nodes=1
#SBATCH --job-name=deepmet_pt7_pubchem_ms2c_formula_pred_arr
#SBATCH --mem=80G
#SBATCH --cpus-per-task=40
#SBATCH --time=0-3:00
#SBATCH --mail-user=fw4@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --output=/home/feiw/feiw/frag-gnn/jobs/deepmet_pt7_ms2c_pred_arr_%A_%a.out
#SBATCH --error=/home/feiw/feiw/frag-gnn/jobs/deepmet_pt7_ms2c_pred_arr_%A_%a.err
#SBATCH --open-mode=append
#SBATCH --array=1-20

# record start
echo `date`: Job ${SLURM_JOB_ID} is allocated resources
#load modules
module load StdEnv/2023
module load python/3.10
module load rdkit/2023.09.5

# go to script dir
cd /home/feiw/feiw/frag-gnn

# copy proc
mkdir -p ${SLURM_TMPDIR}/proc
cp frag-gnn-data/ms2c/deepmet/proc/pubchem_proc/mol_df_${SLURM_ARRAY_TASK_ID}.pkl.gz ${SLURM_TMPDIR}/proc
cp frag-gnn-data/ms2c/deepmet/proc/pubchem_proc/spec_df_${SLURM_ARRAY_TASK_ID}.pkl.gz ${SLURM_TMPDIR}/proc

# copy frags
mkdir -p ${SLURM_TMPDIR}/frags
cp /scratch/frag-gnn-data/ms2c/deepmet/frags/pubchem_d3_h4/${SLURM_ARRAY_TASK_ID}.tar ${SLURM_TMPDIR}
tar -xf ${SLURM_TMPDIR}/${SLURM_ARRAY_TASK_ID}.tar -C ${SLURM_TMPDIR}
mv ${SLURM_TMPDIR}/${SLURM_ARRAY_TASK_ID} ${SLURM_TMPDIR}/frag

# copy apptainer container
module load apptainer/1.2.4
cp ./containers/apptainer/fragnnet_cpu.sif ${SLURM_TMPDIR}/fragnnet_cpu.sif
apptainer exec ${SLURM_TMPDIR}/fragnnet_cpu.sif pip install --user -I -e .

apptainer exec -B ${SLURM_TMPDIR} \
    ${SLURM_TMPDIR}/fragnnet_cpu.sif \
    python ./scripts/ms2c/run_ms2c_predict_deepmet.py \
    --device=cpu \
    --save_fp=${SLURM_TMPDIR}/deepmet_pt7_pubchem_${SLURM_ARRAY_TASK_ID}.pkl \
    --mol_df_fp=${SLURM_TMPDIR}/proc/mol_df_${SLURM_ARRAY_TASK_ID}.pkl.gz \
    --spec_df_fp=${SLURM_TMPDIR}/proc/spec_df_${SLURM_ARRAY_TASK_ID}.pkl.gz \
    --frag_dp=${SLURM_TMPDIR}/frag \
    --use_cahced_ckpt=True
    --model_ckpt=./saved_ckpts/d3_um_prec_pcdl_inchikey_0kxgjooa.ckpt \
    --batch_size=10 \
    --int_min = 10

#tar -cf retrieval_candidates_formula_frags_${SLURM_ARRAY_TASK_ID}.tar retrieval_candidates_formula_frags_${SLURM_ARRAY_TASK_ID}
mkdir -p /scratch/feiw/frag-gnn-data/ms2c/deepmet_pt7_pubchem/fragnnet_predicted/
cp -f ${SLURM_TMPDIR}/deepmet_pt7_pubchem_${SLURM_ARRAY_TASK_ID}.pkl  /scratch/feiw/frag-gnn-data/ms2c/deepmet_pt7_pubchem/fragnnet_predicted/

# record ending
echo `date`: Job ${SLURM_JOB_ID} finished running
