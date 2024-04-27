

for split in ['inchikey','scaffold']:
    for model in ['fraggnn_d3','fraggnn_d4','fraggnn_d3_v2','fraggnn_d4_v2','iceberg','massformer','massformer_v2','neims_v2','neims_v2']:
        print(f"python scripts/ms2c/run_ms2c_grp_stats.py  --per_grp_dir=data/ms2c/pubchem/results/{split}_{model}_pubchem_nist20_d4_wr00_{split}_test --result_csv=data/benchmark/icml_ms2c/{split}/{model}_{split}.csv")