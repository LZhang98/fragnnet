#!/bin/bash
#SBATCH --account=def-someuser
#SBATCH --mem-per-cpu=1.5G      # increase as needed
#SBATCH --time=1:00:00

module load python/3.10
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r ./cc_requirements.txt
pip install -e .