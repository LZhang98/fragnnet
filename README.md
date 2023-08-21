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



## How to run this on ComputeCanada
### Create Env
Run ```env/env_cc_dev_gpu_torch2.sh``` or Create virtualenv via ```env/cc_requirement.txt```

```
 python ./scripts/setup_job_cc.py --python_script_fp=./scripts/run_pl_model_fit.py --job_name=d3_forumla_only --frag_gnn_dp=./ --script_kwargs="w=offline,c=config/d3_forumla_only.yml"
```
### D3 interstage
```
python ./scripts/setup_job_cc.py --python_script_fp=./scripts/run_pl_model_fit.py --job_name=d3_bb_hlr_int_add --frag_gnn_dp=./ --script_kwargs="w=offline,c=config/d3_bb_hlr_int_add.yml" --time=12
python ./scripts/setup_job_cc.py --python_script_fp=./scripts/run_pl_model_fit.py --job_name=d3_bb_hlr_int_direct --frag_gnn_dp=./ --script_kwargs="w=offline,c=config/d3_bb_hlr_int_direct.yml" --time=12
python ./scripts/setup_job_cc.py --python_script_fp=./scripts/run_pl_model_fit.py --job_name=d3_bb_hlr_int_linear --frag_gnn_dp=./ --script_kwargs="w=offline,c=config/d3_bb_hlr_int_linear.yml" --time=12
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
### login to wandb before do anything
[https://docs.wandb.ai/quickstart]
```
wandb login
```