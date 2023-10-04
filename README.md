# frag-gnn
Mass spectrometry prediction using combinatorial fragmentation and graph neural networks.


## Used libs
### SpectralEntropy
Publication [https://doi.org/10.1038/s41592-021-01331-z]
Github ['https://github.com/YuanyueLi/SpectralEntropy']
Doc [https://spectralentropy.readthedocs.io/en/master/]

## For developers
### run memory check
valgrind --tool=memcheck --leak-check=full --suppressions=./data/valgrind-python.supp  --log-file=minimal.valgrind.log python ./sc
ripts/run_compute_frags.py 
### check memory usage
memray run --native ./scripts/run_compute_frags.py

memray flamegraph ./scripts/run_compute_frags.abc.bin


### Profile
[https://medium.com/@narenandu/profiling-and-visualization-tools-in-python-89a46f578989]
~ pip install snakeviz
snakeviz run_frag_exp.

## Preprocessing
### Create frag and features
```
python ./preproc_scripts/03_prepare_dag_feats.py --max_depth=4 --wandb_mode=off --run_name=d4_h4_isoTrue_v2
```

### Create data split use script
Data split should be done respect to different frag size
```
python ./preproc_scripts/04_prepare_split.py --max_num_dag_nodes=100000 --split_name=d4_h4_isoTrue_v2 --frag_name=d4_h4_isoTrue_v2
```

## How to run this on ComputeCanada
### Create Env
Run ```env/env_cc_dev_gpu_torch2.sh``` or Create virtualenv via ```env/cc_requirement.txt```

### d3 forumla only
```
 python ./scripts/setup_job_cc.py --python_script_fp=./scripts/run_pl_model_fit.py --job_name=d3_forumla_only --frag_gnn_dp=./ --script_kwargs="w=offline,c=config/d3_forumla_only.yml"
```

### D3 GAT
```
python ./scripts/setup_job_cc.py --python_script_fp=./scripts/run_pl_model_fit.py --job_name=d3_bb_hlr_gatv2_1head --frag_gnn_dp=./ --script_kwargs="w=offline,c=config/d3_bb_hlr_gatv2_1head.yml" --time=12
```

### D3 interstage
```
python ./scripts/setup_job_cc.py --python_script_fp=./scripts/run_pl_model_fit.py --job_name=d3_bb_hlr_int_add --frag_gnn_dp=./ --script_kwargs="w=offline,c=config/d3_bb_hlr_int_add.yml" --time=12
python ./scripts/setup_job_cc.py --python_script_fp=./scripts/run_pl_model_fit.py --job_name=d3_bb_hlr_int_direct --frag_gnn_dp=./ --script_kwargs="w=offline,c=config/d3_bb_hlr_int_direct.yml" --time=12
python ./scripts/setup_job_cc.py --python_script_fp=./scripts/run_pl_model_fit.py --job_name=d3_bb_hlr_int_linear --frag_gnn_dp=./ --script_kwargs="w=offline,c=config/d3_bb_hlr_int_linear.yml" --time=12
```

### D3 softmax pooling
```
python ./scripts/setup_job_cc.py --python_script_fp=./scripts/run_pl_model_fit.py --job_name=d3_bb_hlr_softxmax_pooling --frag_gnn_dp=./ --script_kwargs="w=offline,c=config/d3_bb_hlr_softxmax_pooling.yml,j=d3_bb_hlr_softxmax_pooling" --time=12
```
### D4 Tasks
```
python ./scripts/setup_job_cc.py --python_script_fp=./scripts/run_pl_model_fit.py --job_name=d4_bb_hlr_negfe --frag_gnn_dp=./ --script_kwargs="w=offline,c=config/d4_bb_hlr_negfe.yml" --time=12
python ./scripts/setup_job_cc.py --python_script_fp=./scripts/run_pl_model_fit.py --job_name=d4_bb_hlr_negne_posnfe --frag_gnn_dp=./ --script_kwargs="w=offline,c=config/d4_bb_hlr_negne_posnfe.yml" --time=12
python ./scripts/setup_job_cc.py --python_script_fp=./scripts/run_pl_model_fit.py --job_name=d4_bb_hlr_negne --frag_gnn_dp=./ --script_kwargs="w=offline,c=config/d4_bb_hlr_negne.yml" --time=12
python ./scripts/setup_job_cc.py --python_script_fp=./scripts/run_pl_model_fit.py --job_name=d4_bb_hlr_negnfe --frag_gnn_dp=./ --script_kwargs="w=offline,c=config/d4_bb_hlr_negnfe.yml" --time=12
python ./scripts/setup_job_cc.py --python_script_fp=./scripts/run_pl_model_fit.py --job_name=d4_bb_hlr_negnne --frag_gnn_dp=./ --script_kwargs="w=offline,c=config/d4_bb_hlr_negnne.yml" --time=12
python ./scripts/setup_job_cc.py --python_script_fp=./scripts/run_pl_model_fit.py --job_name=d4_bb_hlr_nfout --frag_gnn_dp=./ --script_kwargs="w=offline,c=config/d4_bb_hlr_nfout.yml" --time=12
python ./scripts/setup_job_cc.py --python_script_fp=./scripts/run_pl_model_fit.py --job_name=d4_bb_hlr_posfe --frag_gnn_dp=./ --script_kwargs="w=offline,c=config/d4_bb_hlr_posfe.yml" --time=12
python ./scripts/setup_job_cc.py --python_script_fp=./scripts/run_pl_model_fit.py --job_name=d4_bb_hlr_posne_posnfe --frag_gnn_dp=./ --script_kwargs="w=offline,c=config/d4_bb_hlr_posne_posnfe.yml" --time=12
python ./scripts/setup_job_cc.py --python_script_fp=./scripts/run_pl_model_fit.py --job_name=d4_bb_hlr_posne --frag_gnn_dp=./ --script_kwargs="w=offline,c=config/d4_bb_hlr_posne.yml" --time=12
python ./scripts/setup_job_cc.py --python_script_fp=./scripts/run_pl_model_fit.py --job_name=d4_bb_hlr_posnfe_negfe --frag_gnn_dp=./ --script_kwargs="w=offline,c=config/d4_bb_hlr_posnfe_negfe.yml" --time=12
python ./scripts/setup_job_cc.py --python_script_fp=./scripts/run_pl_model_fit.py --job_name=d4_bb_hlr_posnfe --frag_gnn_dp=./ --script_kwargs="w=offline,c=config/d4_bb_hlr_posnfe.yml" --time=12
python ./scripts/setup_job_cc.py --python_script_fp=./scripts/run_pl_model_fit.py --job_name=d4_bb_hlr_posnne --frag_gnn_dp=./ --script_kwargs="w=offline,c=config/d4_bb_hlr_posnne.yml" --time=12
```

### Test sampler
```
python ./scripts/setup_job_cc.py --python_script_fp=./scripts/run_pl_model_fit.py --job_name=d3_base_ds_fp16 --frag_gnn_dp=./ --script_kwargs="w=offline,c=config/samplers_exp/d3_base_ds_fp16.yml,-j=d3_base_ds_fp16" --time=12 --cc_account=rrg-rgreiner
python ./scripts/setup_job_cc.py --python_script_fp=./scripts/run_pl_model_fit.py --job_name=d3_base_ds_fp32 --frag_gnn_dp=./ --script_kwargs="w=offline,c=config/samplers_exp/d3_base_ds_fp32.yml,-j=d3_base_ds_fp32" --time=12 --cc_account=rrg-rgreiner
python ./scripts/setup_job_cc.py --python_script_fp=./scripts/run_pl_model_fit.py --job_name=d3_base_ds --frag_gnn_dp=./ --script_kwargs="w=offline,c=config/samplers_exp/d3_base_ds.yml,-j=d3_base_ds" --time=12 --cc_account=rrg-rgreiner
python ./scripts/setup_job_cc.py --python_script_fp=./scripts/run_pl_model_fit.py --job_name=d4_100K_ds --frag_gnn_dp=./ --script_kwargs="w=offline,c=config/samplers_exp/d4_100K_ds.yml,-j=d4_100K_ds" --time=12 --cc_account=rrg-rgreiner
python ./scripts/setup_job_cc.py --python_script_fp=./scripts/run_pl_model_fit.py --job_name=d4_30K_ds --frag_gnn_dp=./ --script_kwargs="w=offline,c=config/samplers_exp/d4_30K_ds.yml,-j=d4_30K_ds" --time=12 --cc_account=rrg-rgreiner
python ./scripts/setup_job_cc.py --python_script_fp=./scripts/run_pl_model_fit.py --job_name=d4_50K_ds --frag_gnn_dp=./ --script_kwargs="w=offline,c=config/samplers_exp/d4_50K_ds.yml,-j=d4_50K_ds" --time=12 --cc_account=rrg-rgreiner

python ./scripts/setup_job_cc.py --python_script_fp=./scripts/run_pl_model_fit.py --job_name=d3_base_ds_fp32_250K --frag_gnn_dp=./ --script_kwargs="w=offline,c=config/samplers_exp/d3_base_ds_fp32_250K.yml,-j=d4_50K_ds" --time=1 --cc_account=rrg-rgreiner

python ./scripts/setup_job_cc.py --python_script_fp=./scripts/run_pl_model_fit.py --job_name=d4_30K_fp32_250K --frag_gnn_dp=./ --script_kwargs="w=offline,c=config/samplers_exp/d4_30K_fp32_250K.yml,-j=d4_50K_ds" --time=1 --cc_account=rrg-rgreiner
```

### 500K sample
```
python ./scripts/setup_job_cc.py --python_script_fp=./scripts/run_pl_model_fit.py --job_name=d3_base_ds_fp32_500K --frag_gnn_dp=./ --script_kwargs="w=offline,c=config/samplers_exp/ -j d3_base_ds_fp32_500K.yml,-j=d3_base_ds_fp32_500K" --time=1 --cc_account=rrg-rgreiner


python ./scripts/setup_job_cc.py --python_script_fp=./scripts/run_pl_model_fit.py --job_name=d4_30K_fp32_500K --frag_gnn_dp=./ --script_kwargs="w=offline,c=config/samplers_exp/d4_30K_fp32_500K.yml,-j=d4_30K_fp32_500K" --time=1 --cc_account=rrg-rgreiner

python ./scripts/setup_job_cc.py --python_script_fp=./scripts/run_pl_model_fit.py --job_name=d4_50K_fp32_500K --frag_gnn_dp=./ --script_kwargs="w=offline,c=config/samplers_exp/d4_50K_fp32_500K.yml,-j=d4_50K_fp32_500K" --time=1 --cc_account=rrg-rgreiner

python ./scripts/setup_job_cc.py --python_script_fp=./scripts/run_pl_model_fit.py --job_name=d4_100K_fp32_500K --frag_gnn_dp=./ --script_kwargs="w=offline,c=config/samplers_exp/d4_100K_fp32_500K.yml,-j=d4_100K_fp32_500K" --time=1 --cc_account=rrg-rgreiner


```
### login to wandb before do anything
[https://docs.wandb.ai/quickstart]
```
wandb login
```