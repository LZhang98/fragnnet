#!/bin/bash
#SBATCH --account=rrg-dwishart
#SBATCH --nodes=1
#SBATCH --job-name=frag_gen_pubchem_inchikey_d3
#SBATCH --mem=90G
#SBATCH --cpus-per-task=40
#SBATCH --time=0-2:00
#SBATCH --mail-user=fw4@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --output=/home/feiw/feiw/frag-gnn/jobs/frag_gen_pubchem_inchikey_d3_%j.out
#SBATCH --error=/home/feiw/feiw/frag-gnn/jobs/frag_gen_pubchem_inchikey_d3_%j.err
#SBATCH --open-mode=append

# record start
echo `date`: Job ${SLURM_JOB_ID} is allocated resources
#load modules
module load StdEnv/2023
module load python/3.10

# go to script dir
cd /home/feiw/feiw/frag-gnn
# load env
source ./FRAG-GNN-GPU2/bin/activate

python preproc_scripts/tmlr_ms2c/10_run_frag_gen.py \
    --frag_dp=${SLURM_TMPDIR}/pubchem_nist20mona23v3_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50  \
    --proc_dp=data/ms2c/pubchem/pred_inputs/pubchem_nist20mona23v3_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50 \
    --max_depth=3

cd ${SLURM_TMPDIR}
tar -cf pubchem_nist20mona23v3_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50.tar pubchem_nist20mona23v3_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50
cp pubchem_nist20mona23v3_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50.tar /scratch/feiw/frag-gnn-data/ms2c/pubchem/frags/d3_h4_isoFalse/

# record ending
echo `date`: Job ${SLURM_JOB_ID} finished running
