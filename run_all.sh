python scripts/benchmarking/pubchem_v1/run_ms2c_predict_tmlr2.py \
	--save_fp=./data/ms2c/pubchem/predicted/ms2pubchem_nist20mona23v3_d4_wr00_scaffold_test_10ppm_MORGAN-R2_50/fragnnet_d3_s0.pkl \
	--proc_dp=./data/ms2c/pubchem/proc/pubchem_nist20mona23v3_d4_wr00_scaffold_test_10ppm_MORGAN-R2_50 \
	--frag_dp=./data/ms2c/pubchem/frags/d3_h4_isoFalse/pubchem_nist20mona23v3_d4_wr00_scaffold_test_10ppm_MORGAN-R2_50 \
	--wandb_run_id=ad8s7pkj \
    --batch_size=8

python scripts/benchmarking/pubchem_v1/run_ms2c_predict_tmlr2.py \
	--save_fp=./data/ms2c/pubchem/predicted/ms2pubchem_nist20mona23v3_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50/fragnnet_d4_s0.pkl \
	--proc_dp=./data/ms2c/pubchem/proc/pubchem_nist20mona23v3_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50 \
	--frag_dp=./data/ms2c/pubchem/frags/d4_h4_isoFalse/pubchem_nist20mona23v3_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50 \
	--wandb_run_id=a64wr0sb \
    --batch_size=1

python scripts/benchmarking/pubchem_v1/run_ms2c_predict_tmlr2.py \
	--save_fp=./data/ms2c/pubchem/predicted/ms2pubchem_nist20mona23v3_d4_wr00_scaffold_test_10ppm_MORGAN-R2_50/fragnnet_d4_s0.pkl \
	--proc_dp=./data/ms2c/pubchem/proc/pubchem_nist20mona23v3_d4_wr00_scaffold_test_10ppm_MORGAN-R2_50 \
	--frag_dp=./data/ms2c/pubchem/frags/d4_h4_isoFalse/pubchem_nist20mona23v3_d4_wr00_scaffold_test_10ppm_MORGAN-R2_50 \
	--wandb_run_id=wfm2tkrx \
    --batch_size=1
