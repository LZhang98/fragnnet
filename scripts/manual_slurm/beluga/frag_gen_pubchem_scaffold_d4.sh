#!/bin/bash
#SBATCH --account=rrg-dwishart
#SBATCH --nodes=1
#SBATCH --job-name=frag_gen_pubchem_scaffold_d4
#SBATCH --mem=90G
#SBATCH --cpus-per-task=40
#SBATCH --time=0-6:00
#SBATCH --mail-user=fw4@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --output=/home/feiw/feiw/frag-gnn/jobs/frag_gen_pubchem_scaffold_d4%j.out
#SBATCH --error=/home/feiw/feiw/frag-gnn/jobs/frag_gen_pubchem_scaffold_d4%j.err
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

python preproc_scripts/pubchem_ms2c/10_run_ms2c_frag_gen.py \
    --frag_dp=${SLURM_TMPDIR}/pubchem_nist20mona23v3_d4_wr00_scaffold_test_10ppm_MORGAN-R2_50  \
    --proc_dp=data/ms2c/pubchem/proc/pubchem_nist20mona23v3_d4_wr00_scaffold_test_10ppm_MORGAN-R2_50 \
    --max_depth=4

cd ${SLURM_TMPDIR}
tar -cf pubchem_nist20mona23v3_d4_wr00_scaffold_test_10ppm_MORGAN-R2_50.tar pubchem_nist20mona23v3_d4_wr00_scaffold_test_10ppm_MORGAN-R2_50

mkdir -p /scratch/feiw/frag-gnn-data/ms2c/pubchem/frags/d4_h4_isoFalse/
cp pubchem_nist20mona23v3_d4_wr00_scaffold_test_10ppm_MORGAN-R2_50.tar /scratch/feiw/frag-gnn-data/ms2c/pubchem/frags/d4_h4_isoFalse/

# record ending
echo `date`: Job ${SLURM_JOB_ID} finished running
