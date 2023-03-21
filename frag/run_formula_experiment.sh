#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --account=rrg-hroest
#SBATCH --mem=250000M
#SBATCH --cpus-per-task=8
#SBATCH --time=0-12:00:00
#SBATCH --job-name=formula_exp
#SBATCH --output=/home/youngad2/projects/rrg-hroest/youngad2/jobs/%j.out
#SBATCH --error=/home/youngad2/projects/rrg-hroest/youngad2/jobs/%j.out
#SBATCH --open-mode=append

set -e

. /home/youngad2/.bashrc

module load StdEnv/2020 gcc/9.3.0 python/3.10 rdkit/2022.09.4 cuda/11.1.1 rust/1.65.0

source /home/youngad2/projects/rrg-hroest/youngad2/GFN/bin/activate

cd /home/youngad2/projects/rrg-hroest/youngad2/comb-gnn/frag

python formula_experiment.py
