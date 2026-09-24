#!/usr/bin/env bash
set -euo pipefail

# Step 1: generate fragmentation DAGs (depth 3) for the FT + QTOF molecule set.
# The existing dags (data/frag/nist20_d3) were computed on FT spectra only, so
# QTOF-only molecules have no dag and would break training. Writing to a new
# frag dir keeps the FT-only artifacts untouched.
python preproc_scripts/03_prepare_dag_feats.py \
	--max_depth 3 \
	--frag_dp data/frag/nist20_ft_qtof_d3 \
	--proc_dp data/proc/nist20 \
	--prec_types "[M+H]+" \
	--inst_types FT QTOF \
	--ion_modes P \
	--dsets nist20_hr \
	--wandb_mode disabled \
	--nb_isomorphic False

# Step 2: generate the scaffold split over the FT + QTOF spectra.
# dag_filtering keeps only molecules that have a dag in the new frag dir.
python preproc_scripts/04_prepare_split.py \
	--split_key scaffold \
	--primary_dsets nist20_hr \
	--prec_types "[M+H]+" \
	--inst_types FT QTOF \
	--ion_modes P \
	--ces any \
	--max_peak_mz 1500.0 \
	--max_prec_mz 1500.0 \
	--min_prec_mz 0.0 \
	--dag_filtering True \
	--dag_filter_grouping m_spec \
	--max_num_dag_nodes 100000 \
	--max_num_dag_edges 250000 \
	--min_dag_wrecall 0.00 \
	--proc_dp data/proc/nist20 \
	--frag_dp data/frag/nist20_ft_qtof_d3 \
	--split_dp data/split/nist20_ft_qtof_scaffold