#!/bin/bash
#SBATCH --account=rrg-dwishart
#SBATCH --nodes=1
#SBATCH --job-name=msgym_ms2c_formula_pred_arr
#SBATCH --mem=80G
#SBATCH --cpus-per-task=40
#SBATCH --time=0-3:00
#SBATCH --mail-user=fw4@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --output=/home/feiw/feiw/frag-gnn/jobs/msgym_ms2c_formula_pred_arr%j.out
#SBATCH --error=/home/feiw/feiw/frag-gnn/jobs/msgym_ms2c_formula_pred_arr%j.err
#SBATCH --open-mode=append
#SBATCH --array=0-10000:400

# record start
echo `date`: Job ${SLURM_JOB_ID} is allocated resources
#load modules
module load StdEnv/2023
module load python/3.10

# go to script dir
cd /home/feiw/feiw/frag-gnn
# copy proc
cp /scratch/feiw/frag-gnn-data/ms2c/msgym/proc/retrieval_candidates_formula.tar ${SLURM_TMPDIR}/retrieval_candidates_formula.tar
tar -xf ${SLURM_TMPDIR}/retrieval_candidates_formula.tar -C ${SLURM_TMPDIR}
mv ${SLURM_TMPDIR}/retrieval_candidates_formula ${SLURM_TMPDIR}/proc
# copy frags
cp /scratch/feiw/frag-gnn-data/ms2c/msgym/frags/retrieval_candidates_formula_d3_h4.tar ${SLURM_TMPDIR}
tar -xf ${SLURM_TMPDIR}/retrieval_candidates_formula_d3_h4.tar -C ${SLURM_TMPDIR}
mv ${SLURM_TMPDIR}/retrieval_candidates_formula_d3_h4 ${SLURM_TMPDIR}/frag

# copy apptainer container
module load apptainer/1.2.4
cp ./containers/apptainer/fragnnet_cpu.sif ${SLURM_TMPDIR}/fragnnet_cpu.sif
apptainer exec ${SLURM_TMPDIR}/fragnnet_cpu.sif pip install --user -I -e .

apptainer exec -B ${SLURM_TMPDIR} \
    ${SLURM_TMPDIR}/fragnnet_cpu.sif \
    python ./scripts/ms2c/run_ms2c_predict_msgym.py \
    --device=cpu \
    --save_dp=${SLURM_TMPDIR}/msgym/fragnnet_predicted/retrieval_candidates_formula \
    --proc_dp=${SLURM_TMPDIR}/proc \
    --frag_dp=${SLURM_TMPDIR}/frag \
    --zipped_frag=False \
    --model_ckpt=./saved_ckpts/msg_fragnnet_0,01_v2_9gnx820u/msg_fragnnet_0,01_v2_9gnx820u.ckpt \
    --start_row_idx=${SLURM_ARRAY_TASK_ID} \
	--num_row=400

#tar -cf retrieval_candidates_formula_frags_${SLURM_ARRAY_TASK_ID}.tar retrieval_candidates_formula_frags_${SLURM_ARRAY_TASK_ID}
mkdir -p /scratch/feiw/frag-gnn-data/ms2c/msgym/fragnnet_predicted/retrieval_candidates_formula/
cp -f ${SLURM_TMPDIR}/msgym/fragnnet_predicted/retrieval_candidates_formula/* /scratch/feiw/frag-gnn-data/ms2c/msgym/fragnnet_predicted/retrieval_candidates_formula/

# record ending
echo `date`: Job ${SLURM_JOB_ID} finished running
