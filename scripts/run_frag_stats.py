
import pandas as pd
import torch as th
import os
from tqdm import tqdm

from frag_gnn.runner import load_config, init_dataset, init_dataloader
from frag_gnn.pl_model import FragGNNPL
from frag_gnn.utils.misc_utils import to_device

if __name__ == "__main__":

    split_dp = "data/split/nist20mona23v3_d4_wr00_inchikey"
    template_config_fp = "config/template.yml"
    d3_config_fp = "config/frag_stats/d3.yml"
    d4_config_fp = "config/frag_stats/d4.yml"

    d3_output_fp = "figs/frag_stats/d3.csv"
    d4_output_fp = "figs/frag_stats/d4.csv"

    def get_results(depth):
        if depth == "d3":
            config_fp = d3_config_fp
        elif depth == "d4":
            config_fp = d4_config_fp
        print(">> Load config")
        config = load_config(template_config_fp,config_fp)
        print(">> Init dataset")
        ds = init_dataset(config, splits=("all",))[0]
        print(">> Init dataloader")
        dl = init_dataloader(ds, config)
        print(">> Init model")
        model = FragGNNPL(**config)
        device = th.device("cuda:0")
        model = model.to(device)
        model = model.eval()
        # do inference
        print(">> Inference")
        results = {
            "recall": [],
            "wrecall": [],
            "precision": [],
            "wprecision": [],
            "pred_node_count": [],
            "pred_nb_node_count": [],
            "pred_formula_count": [],
            "pred_edge_count": [],
        }
        with th.inference_mode():
            for b_idx, b_input in tqdm(enumerate(dl),total=len(dl)):
                b_input = to_device(b_input, device, non_blocking=False)
                b_output = model.inference_step(
                    b_input,
                    split="test"
                )
                b_output = to_device(b_output, "cpu", non_blocking=False)
                for k in results.keys():
                    results[k].append(b_output[k])
        results = {k: th.cat(v, dim=0) for k, v in results.items()}
        return results
    
    # D3
    d3_results = get_results("d3")
    d3_results_df = pd.DataFrame(d3_results)
    d3_results_df.to_csv(d3_output_fp, index=False)
    del d3_results

    # D4
    d4_results = get_results("d4")
    d4_results_df = pd.DataFrame(d4_results)
    d4_results_df.to_csv(d4_output_fp, index=False)
    del d4_results
