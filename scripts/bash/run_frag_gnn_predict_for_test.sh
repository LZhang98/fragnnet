#!/bin/bash
ulimit -n 2048
# inchikey d4
python ./scripts/misc/run_frag_gnn_test.py  \
--custom_fp="config/nist20_d4_wr00_inchikey/fraggnn_d4_nodemlp_mnce_nocp.yml" \
--ckpt_fp="./saved_ckpts/kme8fry7_fraggnn_d4_nodemlp_mnce_nocp/ckpt/model-epoch=098.ckpt" \
--save_fp="./data/predicted/nist20_d4_wr00_inchikey/kme8fry7_fraggnn_d4_nodemlp_mnce_nocp.pkl.gz"

# scaffold d4
python ./scripts/misc/run_frag_gnn_test.py \
--custom_fp="config/nist20_d4_wr00_scaffold/fraggnn_d4_nodemlp_mnce_nocp.yml" \
--ckpt_fp="./saved_ckpts/gmpfciq9_fraggnn_d4_nodemlp_mnce_nocp/ckpt/model-epoch=096.ckpt" \
--save_fp="./data/predicted/nist20_d4_wr00_scaffold/gmpfciq9_fraggnn_d4_nodemlp_mnce_nocp.pkl.gz"

# inchikey d3
python ./scripts/misc/run_frag_gnn_test.py \
--custom_fp="config/nist20_d4_wr00_inchikey/fraggnn_d3_nodemlp_mnce_nocp.yml" \
--ckpt_fp="saved_ckpts/ohtiivyj_fraggnn_d3_nodemlp_mnce_nocp/ckpt/model-epoch=093.ckpt" \
--save_fp="./data/predicted/nist20_d4_wr00_inchikey/ohtiivyj_fraggnn_d3_nodemlp_mnce_nocp.pkl.gz"
    
# scaffold d3
python ./scripts/misc/run_frag_gnn_test.py \
--custom_fp="config/nist20_d4_wr00_scaffold/fraggnn_d3_nodemlp_mnce_nocp.yml" \
--ckpt_fp="saved_ckpts/lgeo7mjz_fraggnn_d3_nodemlp_mnce_nocp/ckpt/model-epoch=089.ckpt" \
--save_fp="./data/predicted/nist20_d4_wr00_scaffold/lgeo7mjz_fraggnn_d3_nodemlp_mnce.pkl.gz"