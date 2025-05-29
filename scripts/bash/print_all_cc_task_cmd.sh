#!/bin/bash

# Check if directory and account arguments are provided
if [ -z "$1" ] || [ -z "$2" ]||[ -z "$3"]; then
    echo "Usage: $0 <config_directory> <cc_account> <wall_time>"
    exit 1
fi

config_dir="$1"
cc_account="$2"
wall_time="$3"

# Check if there are any .yml files in the directory
shopt -s nullglob
yml_files=("$config_dir"/*.yml)
if [ ${#yml_files[@]} -eq 0 ]; then
    echo "No .yml files found in directory: $config_dir"
    exit 1
fi

for fullfile in "${yml_files[@]}"; do
    filename="$(basename -- "$fullfile")"
    filename="${filename%.*}"  # Remove extension

    echo "python ./scripts/setup_job_cc.py \\
        --python_script_fp=./scripts/run_pl_model_fit.py \\
        --job_name=${filename} \\
        --frag_gnn_dp=./ \\
        --script_kwargs w=offline c=${config_dir}/${filename}.yml \\
        --cc_account=${cc_account} \\
        --time=${wall_time}"
done