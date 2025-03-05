#s0 
cd /home/feiw/gitRepo/frag-gnn
cd ./data/ms2c/pubchem/magma
tar xvf pred_magma_nist20v3_scaffold_iceberg_gen_s0_4pweeg45.tar
cd /home/feiw/gitRepo/frag-gnn
python scripts/ms2c/pubchem_v1/run_iceberg_ms2c_predict_val.py \
	--iceberg_inten_batch_size=128 \
	--proc_dp=./data/ms2c/pubchem/proc/pubchem_nist20mona23v3_d4_wr00_scaffold_test_10ppm_MORGAN-R2_50 \
	--save_fp=./data/ms2c/pubchem/predicted/ms2pubchem_nist20mona23v3_d4_wr00_scaffold_test_10ppm_MORGAN-R2_50/iceberg_s0.pkl \
	--magma_dp=./data/ms2c/pubchem/magma/pred_magma_nist20v3_scaffold_iceberg_gen_s0_4pweeg45 \
	--iceberg_inten_wandb_run_id=riyr41po

#cd ./data/ms2c/pubchem/magma
#rm -rf pred_magma_nist20v3_scaffold_iceberg_gen_s0_4pweeg45

#s1
#cd /home/feiw/gitRepo/frag-gnn
#cd ./data/ms2c/pubchem/magma
#tar xvf pred_magma_nist20v3_scaffold_iceberg_gen_s1_9j26mr4s.tar
#cd /home/feiw/gitRepo/frag-gnn
#python scripts/ms2c/pubchem_v1/run_iceberg_ms2c_predict_val.py \
#	--iceberg_inten_batch_size=128 \
#	--run_iceberg_chunk_size=50000 \
#	--save_fp=./data/ms2c/pubchem/predicted/ms2pubchem_nist20mona23v3_d4_wr00_scaffold_test_10ppm_MORGAN-R2_50/iceberg_s1.pkl\
#	--magma_dp=./data/ms2c/pubchem/magma/pred_magma_nist20v3_scaffold_iceberg_gen_s1_9j26mr4s \
#	--iceberg_inten_wandb_run_id=pmaom333
#cd ./data/ms2c/pubchem/magma
#rm -rf pred_magma_nist20v3_scaffold_iceberg_gen_s1_9j26mr4s

# s2
#cd /home/feiw/gitRepo/frag-gnn
#cd ./data/ms2c/pubchem/magma
#tar xvf pred_magma_nist20v3_scaffold_iceberg_gen_s2_n27tube2.tar
#cd /home/feiw/gitRepo/frag-gnn
#python scripts/ms2c/pubchem_v1/run_iceberg_ms2c_predict_val.py \
#	--iceberg_inten_batch_size=128 \
#	--run_iceberg_chunk_size=50000 \
#	--save_fp=./data/ms2c/pubchem/predicted/ms2pubchem_nist20mona23v3_d4_wr00_scaffold_test_10ppm_MORGAN-R2_50/iceberg_s2.pkl\
#	--magma_dp=./data/ms2c/pubchem/magma/pred_magma_nist20v3_scaffold_iceberg_gen_s2_n27tube2 \
#	--iceberg_inten_wandb_run_id=s37eax0j
#cd ./data/ms2c/pubchem/magma
#rm -rf pred_magma_nist20v3_scaffold_iceberg_gen_s2_n27tube2

# s3
cd /home/feiw/gitRepo/frag-gnn
cd ./data/ms2c/pubchem/magma
tar xvf pred_magma_nist20v3_scaffold_iceberg_gen_s3_biofdpxl.tar
cd /home/feiw/gitRepo/frag-gnn

python scripts/ms2c/pubchem_v1/run_iceberg_ms2c_predict_val.py \
	--iceberg_inten_batch_size=128 \
	--run_iceberg_chunk_size=50000 \
	--proc_dp=./data/ms2c/pubchem/proc/pubchem_nist20mona23v3_d4_wr00_scaffold_test_10ppm_MORGAN-R2_50 \
	--save_fp=./data/ms2c/pubchem/predicted/ms2pubchem_nist20mona23v3_d4_wr00_scaffold_test_10ppm_MORGAN-R2_50/iceberg_s3.pkl\
	--magma_dp=./data/ms2c/pubchem/magma/pred_magma_nist20v3_scaffold_iceberg_gen_s3_biofdpxl \
	--iceberg_inten_wandb_run_id=x2l4adze

cd ./data/ms2c/pubchem/magma
rm -rf pred_magma_nist20v3_scaffold_iceberg_gen_s3_biofdpxl

# s4
cd /home/feiw/gitRepo/frag-gnn
cd ./data/ms2c/pubchem/magma
tar xvf pred_magma_nist20v3_scaffold_iceberg_gen_s4_cyot56r3.tar
cd /home/feiw/gitRepo/frag-gnn

python scripts/ms2c/pubchem_v1/run_iceberg_ms2c_predict_val.py \
	--iceberg_inten_batch_size=128 \
	--run_iceberg_chunk_size=50000 \
	--proc_dp=./data/ms2c/pubchem/proc/pubchem_nist20mona23v3_d4_wr00_scaffold_test_10ppm_MORGAN-R2_50 \
	--save_fp=./data/ms2c/pubchem/predicted/ms2pubchem_nist20mona23v3_d4_wr00_scaffold_test_10ppm_MORGAN-R2_50/iceberg_s4.pkl\
	--magma_dp=./data/ms2c/pubchem/magma/pred_magma_nist20v3_scaffold_iceberg_gen_s4_cyot56r3 \
	--iceberg_inten_wandb_run_id=6w68jgf6
cd ./data/ms2c/pubchem/magma
rm -rf pred_magma_nist20v3_scaffold_iceberg_gen_s4_cyot56r3