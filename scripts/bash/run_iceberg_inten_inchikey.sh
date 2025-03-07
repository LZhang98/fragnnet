#s0
cd /home/feiw/gitRepo/frag-gnn
cd ./data/ms2c/pubchem/magma
tar xvf pred_magma_nist20v3_inchikey_iceberg_gen_s0_i7k9gc3d.tar
cd /home/feiw/gitRepo/frag-gnn
python scripts/ms2c/pubchem_v1/run_iceberg_ms2c_predict_val.py \
	--iceberg_inten_batch_size=128 \
	--proc_dp=./data/ms2c/pubchem/proc/pubchem_nist20mona23v3_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50 \
	--save_fp=./data/ms2c/pubchem/predicted/ms2pubchem_nist20mona23v3_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50/iceberg_s0.pkl\
	--magma_dp=./data/ms2c/pubchem/magma/pred_magma_nist20v3_inchikey_iceberg_gen_s0_i7k9gc3d \
	--iceberg_inten_wandb_run_id=c8iwu7il
cd ./data/ms2c/pubchem/magma
rm -rf pred_magma_nist20v3_inchikey_iceberg_gen_s0_i7k9gc3d

#s1 
cd /home/feiw/gitRepo/frag-gnn
cd ./data/ms2c/pubchem/magma
tar xvf pred_magma_nist20v3_inchikey_iceberg_gen_s1_mz3s3h62.tar
cd /home/feiw/gitRepo/frag-gnn
python scripts/ms2c/pubchem_v1/run_iceberg_ms2c_predict_val.py \
	--iceberg_inten_batch_size=128 \
	--run_iceberg_chunk_size=50000 \
	--proc_dp=./data/ms2c/pubchem/proc/pubchem_nist20mona23v3_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50 \
	--save_fp=./data/ms2c/pubchem/predicted/ms2pubchem_nist20mona23v3_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50/iceberg_s1.pkl\
	--magma_dp=./data/ms2c/pubchem/magma/pred_magma_nist20v3_inchikey_iceberg_gen_s1_mz3s3h62 \
	--iceberg_inten_wandb_run_id=rtt8i90g
cd ./data/ms2c/pubchem/magma
rm -rf pred_magma_nist20v3_inchikey_iceberg_gen_s1_mz3s3h62

# s2
cd /home/feiw/gitRepo/frag-gnn
cd ./data/ms2c/pubchem/magma
tar xvf pred_magma_nist20v3_inchikey_iceberg_gen_s2_6f4nes2g.tar
cd /home/feiw/gitRepo/frag-gnn
python scripts/ms2c/pubchem_v1/run_iceberg_ms2c_predict_val.py \
	--iceberg_inten_batch_size=128 \
	--run_iceberg_chunk_size=50000 \
	--proc_dp=./data/ms2c/pubchem/proc/pubchem_nist20mona23v3_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50 \
	--save_fp=./data/ms2c/pubchem/predicted/ms2pubchem_nist20mona23v3_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50/iceberg_s2.pkl\
	--magma_dp=./data/ms2c/pubchem/magma/pred_magma_nist20v3_inchikey_iceberg_gen_s2_6f4nes2g \
	--iceberg_inten_wandb_run_id=acgyna2a
cd ./data/ms2c/pubchem/magma
rm -rf pred_magma_nist20v3_inchikey_iceberg_gen_s2_6f4nes2g

# s3
cd /home/feiw/gitRepo/frag-gnn
cd ./data/ms2c/pubchem/magma
tar xvf pred_magma_nist20v3_inchikey_iceberg_gen_s3_lp60e1nq.tar
cd /home/feiw/gitRepo/frag-gnn

python scripts/ms2c/pubchem_v1/run_iceberg_ms2c_predict_val.py \
	--iceberg_inten_batch_size=128 \
	--run_iceberg_chunk_size=50000 \
	--proc_dp=./data/ms2c/pubchem/proc/pubchem_nist20mona23v3_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50 \
	--save_fp=./data/ms2c/pubchem/predicted/ms2pubchem_nist20mona23v3_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50/iceberg_s3.pkl\
	--magma_dp=./data/ms2c/pubchem/magma/pred_magma_nist20v3_inchikey_iceberg_gen_s3_lp60e1nq \
	--iceberg_inten_wandb_run_id=683gijym

cd ./data/ms2c/pubchem/magma
rm -rf pred_magma_nist20v3_inchikey_iceberg_gen_s3_lp60e1nq

# s4
cd /home/feiw/gitRepo/frag-gnn
cd ./data/ms2c/pubchem/magma
tar xvf pred_magma_nist20v3_inchikey_iceberg_gen_s4_3zjhzrss.tar
cd /home/feiw/gitRepo/frag-gnn

python scripts/ms2c/pubchem_v1/run_iceberg_ms2c_predict_val.py \
	--iceberg_inten_batch_size=128 \
	--run_iceberg_chunk_size=50000 \
	--proc_dp=./data/ms2c/pubchem/proc/pubchem_nist20mona23v3_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50 \
	--save_fp=./data/ms2c/pubchem/predicted/ms2pubchem_nist20mona23v3_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50/iceberg_s4.pkl\
	--magma_dp=./data/ms2c/pubchem/magma/pred_magma_nist20v3_inchikey_iceberg_gen_s4_3zjhzrss \
	--iceberg_inten_wandb_run_id=yeatgz7o
cd ./data/ms2c/pubchem/magma
rm -rf pred_magma_nist20v3_inchikey_iceberg_gen_s4_3zjhzrss