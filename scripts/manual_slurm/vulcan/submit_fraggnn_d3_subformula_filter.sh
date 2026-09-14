#!/bin/bash
# Generate (but do not submit) the Vulcan SLURM job for the fraggnn_d3 subformula-filter baseline.

python scripts/setup_job_cc.py \
  --job_name fraggnn_d3_subformula_filter \
  --python_script_fp scripts/run_pl_model_fit.py \
  --frag_gnn_dp . \
  --gpu_profile vulcan:l40s \
  --script_kwargs w=online c=config/exp_fraggnn_ma_mi/fraggnn_d3_ma_mi_nist20_amolc_all_subformula_filter.yml \
  --time 12:00:00 \
  --cc_account aip-rgreiner
