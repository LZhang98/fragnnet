import argparse
from pprint import pprint

import torch as th

import wandb
from fragnnet.dataset import SpecMolFragDataset
from fragnnet.pl_model import FraGNNetPL
from fragnnet.runner import load_config
from fragnnet.utils.misc_utils import NestedDefaultDict, booltype
from fragnnet.utils.script_utils import (
    apply_inference_config,
    cleanup_model,
    init_wandb_ckpt,
    prepare_output_subset,
    run_inference_pkl,
    save_inference_results,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_split", type=str, default="test", choices=["train", "val", "test", "secondary"]
    )
    parser.add_argument("--num_seeds", type=int, default=5)
    parser.add_argument("--save_dp", type=str, required=True)
    parser.add_argument("--force_overwrite", type=booltype, default=False)
    parser.add_argument(
        "--auxiliary_scores",
        type=str,
        nargs="+",
        default=["cos_sim", "cos_hun", "true_oos_prob", "true_oos_e"],
    )
    parser.add_argument("--skip_extra_losses", type=booltype, default=False)
    parser.add_argument(
        "--output_subset",
        type=str,
        nargs="+",
        default=[
            "true_mzs",
            "true_logprobs",
            "true_batch_idxs",
            "pred_mzs",
            "pred_logprobs",
            "pred_batch_idxs",
            "true_oos_prob",
            "oos_prob",
        ],
    )
    parser.add_argument("--output_everything", type=booltype, default=False)
    parser.add_argument("--untransform_spec", type=booltype, default=False)
    parser.add_argument("--disable_preproc", type=booltype, default=True)
    parser.add_argument("--ablation_dset", type=str, default="ablations")
    parser.add_argument(
        "--ablation_model_types",
        type=str,
        nargs="+",
        default=[
            "fragnnet_d3_edges",
            "fragnnet_d3_noce",
            "fragnnet_d4_noce",
        ],
    )
    args = parser.parse_args()

    eval_split = args.eval_split
    num_seeds = args.num_seeds
    save_dp = args.save_dp
    force_overwrite = args.force_overwrite
    ablation_model_types = args.ablation_model_types
    ablation_dset = args.ablation_dset

    # Prepare output subset
    output_subset = prepare_output_subset(
        output_everything=args.output_everything,
        output_subset=set(args.output_subset) if not args.output_everything else None,
    )

    model_type_to_wandb_prefix = {
        "fragnnet_d3_noce": "fraggnn_d3_noce",
        "fragnnet_d3_edges": "fraggnn_d3_edges",
        "fragnnet_d4_noce": "fraggnn_d4_noce",
    }

    dset_to_wandb_prefix = {"ablations": "ablationsv3-1"}

    # this is necessary to avoid a pytorch compile issue...
    # TODO: replace this with a proper checkpoint decompilation
    # th._dynamo.config.suppress_errors = True

    batch_cutoff = int(1e6)

    model_to_run_id = NestedDefaultDict()
    model_to_config = NestedDefaultDict()

    # find wandb checkpoints
    template_fp = "config/template.yml"
    api = wandb.Api()
    for ablation_model_type in ablation_model_types:
        model_to_config[ablation_model_type] = (
            f"config/{ablation_dset}/{ablation_model_type}/s0.yml"
        )
        model_type_wandb_prefix = model_type_to_wandb_prefix[ablation_model_type]
        dset_wandb_group_prefix = dset_to_wandb_prefix[ablation_dset]
        group_name = f"{dset_wandb_group_prefix}_{model_type_wandb_prefix}"
        runs = api.runs(path="frag-gnn/frag-gnn", filters={"group": group_name})
        for run in runs:
            seed = int(run.name.split("_")[-1].removeprefix("s"))
            if seed < num_seeds:
                model_to_run_id[ablation_model_type][seed] = run.id
    del api
    pprint(model_to_config)
    pprint(model_to_run_id)

    for ablation_model_type in ablation_model_types:
        print(f">>> Starting model_type {ablation_model_type}")
        custom_fp = model_to_config[ablation_model_type]

        # Setup configuration
        config_d = load_config(template_fp, custom_fp)
        config_d = apply_inference_config(
            config_d,
            auxiliary_scores=args.auxiliary_scores,
            skip_extra_losses=args.skip_extra_losses,
            disable_preproc=args.disable_preproc,
            output_formula_str=False,
            model_type=ablation_model_type,
        )

        print(">>> Loading dataset")
        ds_cls = SpecMolFragDataset
        from fragnnet.runner import init_dataloader

        ds = ds_cls(split=eval_split, **config_d)
        dl = init_dataloader(ds, config_d)
        model_cls = FraGNNetPL
        device = th.device("cuda:0") if config_d["accelerator"] == "gpu" else th.device("cpu")
        for seed in range(num_seeds):
            print(f">>> Starting seed {seed}")
            run_id = model_to_run_id[ablation_model_type][seed]
            model = init_wandb_ckpt(run_id, last_ckpt=False, model_cls=model_cls, config_d=config_d)
            model.to(device)
            model.eval()

            # Prepare output subset with model-specific metrics
            # output_subset = prepare_output_subset(
            #   output_everything=args.output_everything,
            #   output_subset=set(args.output_subset) if not args.output_everything else None,
            #   model_metric_names=model.metric_names,
            # )

            # run inference
            vals = run_inference_pkl(
                dl,
                model,
                device,
                eval_split,
                batch_cutoff,
                config_d["nb_iso"],
                output_set=output_subset,
            )

            # Save results
            save_inference_results(
                vals=vals,
                save_dp=save_dp,
                model_type=ablation_model_type,
                seed=seed,
                eval_split=eval_split,
                force_overwrite=force_overwrite,
            )

            # Cleanup
            cleanup_model(model, use_gpu=(config_d["accelerator"] == "gpu"))
