#!/bin/bash
#SBATCH --account=rrg-dwishart
#SBATCH --nodes=1
#SBATCH --job-name=msgym_mass_frag_gen_arr
#SBATCH --mem=80G
#SBATCH --cpus-per-task=40
#SBATCH --time=0-3:00
#SBATCH --mail-user=fw4@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --output=/home/feiw/feiw/frag-gnn/jobs/msgym_mass_frag_gen_arr%j.out
#SBATCH --error=/home/feiw/feiw/frag-gnn/jobs/msgym_mass_frag_gen_arr%j.err
#SBATCH --open-mode=append
#SBATCH --array=0-2000:500

# record start
echo `date`: Job ${SLURM_JOB_ID} is allocated resources
#load modules
module load StdEnv/2023
module load python/3.10

# go to script dir
cd /home/feiw/feiw/frag-gnn
# load env
source ./FRAG-GNN-GPU2/bin/activate

cp /scratch/feiw/frag-gnn-data/ms2c/msgym/proc/retrieval_candidates_mass.tar ${SLURM_TMPDIR}/retrieval_candidates_mass.tar
tar -xf ${SLURM_TMPDIR}/retrieval_candidates_mass.tar -C ${SLURM_TMPDIR}

python preproc_scripts/msgym_ms2c/02_preapre_mgsgym_prediction_frag.py \
    --msgym_moldf_dp=${SLURM_TMPDIR}/retrieval_candidates_mass \
    --frag_dp=${SLURM_TMPDIR}/retrieval_candidates_mass_frags_${SLURM_ARRAY_TASK_ID} \
    --max_depth=3 \
    --star_mol_df_id=${SLURM_ARRAY_TASK_ID} \
	--num_ids=500

cd ${SLURM_TMPDIR}
tar -cf retrieval_candidates_mass_frags_${SLURM_ARRAY_TASK_ID}.tar retrieval_candidates_mass_frags_${SLURM_ARRAY_TASK_ID}
mkdir -p /scratch/feiw/frag-gnn-data/ms2c/msgym/frags/retrieval_candidates_mass_d3_h4/
cp retrieval_candidates_mass_frags_${SLURM_ARRAY_TASK_ID}.tar /scratch/feiw/frag-gnn-data/ms2c/msgym/frags/retrieval_candidates_mass_d3_h4/

# record ending
echo `date`: Job ${SLURM_JOB_ID} finished running
