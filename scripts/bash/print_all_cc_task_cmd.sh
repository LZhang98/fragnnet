#!/bin/bash

# Check if directory argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <config_directory>"
    exit 1
fi

config_dir="$1"

for fullfile in "$config_dir"/*.yml; do
    # Check if any .yml files exist
    if [ ! -e "$fullfile" ]; then
        echo "No .yml files found in directory: $config_dir"
        exit 1
    fi

    filename="$(basename -- "$fullfile")"
    filename="${filename%.*}"  # Remove extension

    echo "python ./scripts/setup_job_cc.py \
        --python_script_fp=./scripts/run_pl_model_fit.py \
        --job_name=${filename} \
        --frag_gnn_dp=./ \
        --script_kwargs=\"w=offline,c=${config_dir}/${filename}.yml,-j=${filename}\" \
        --time=12"
done
