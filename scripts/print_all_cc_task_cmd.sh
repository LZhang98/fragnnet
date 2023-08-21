

#!/bin/bash
for fullfile in ./config/*.yml; do
    filename=$(basename -- "$fullfile")
    extension="${filename##*.}"
    filename="${filename%.*}"
    #echo $filename
    echo 'python ./scripts/setup_job_cc.py --python_script_fp=./scripts/run_pl_model_fit.py --job_name='$filename' --frag_gnn_dp=./ --script_kwargs="w=offline,c=config/'$filename'.yml" --time=12'
done