import argparse
from pprint import pprint

import torch as th

import wandb
from fragnnet.dataset import SpecMolDataset, SpecMolFragDataset
from fragnnet.graff.dataset import SpecMolAnnDataset
from fragnnet.graff.pl_model import GrAFFPL
from fragnnet.iceberg.dataset import SpecMolMagmaIntenDataset
from fragnnet.iceberg.pl_model import IcebergIntenPL
from fragnnet.massformer.pl_model import MassFormerPL
from fragnnet.pl_model import FraGNNetPL, NeimsPL, PrecursorPL
from fragnnet.runner import init_dataloader, load_config
from fragnnet.utils.misc_utils import NestedDefaultDict, booltype
from fragnnet.utils.script_utils import (
    apply_inference_config,
    cleanup_model,
    init_wandb_ckpt,
    prepare_output_subset,
    run_inference_tabular,
    save_inference_results,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_types", type=str, nargs="+", default=["inchikey", "scaffold"])
    parser.add_argument(
        "--wandb_model_types",
        type=str,
        nargs="+",
        default=[
            "fragnnet_d3",
            "fragnnet_d4",
            "iceberg_inten",
            "iceberg_inten_opt",
            "neims",
            "massformer",
            "graff",
            "precursor",
        ],
    )
    parser.add_argument("--local_model_types", type=str, nargs="+", default=[])
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
    parser.add_argument("--dset", type=str, default="nist20v3-1")
    args = parser.parse_args()

    eval_split = args.eval_split
    num_seeds = args.num_seeds
    save_dp = args.save_dp
    force_overwrite = args.force_overwrite
    dset = args.dset
    auxiliary_scores = args.auxiliary_scores
    skip_extra_losses = args.skip_extra_losses
    disable_preproc = args.disable_preproc
    output_everything = args.output_everything
    untransform_spec = args.untransform_spec

    # Prepare output subset
    output_subset = prepare_output_subset(
        output_everything=output_everything,
        output_subset=set(args.output_subset) if not output_everything else None,
    )

    # this is necessary to avoid a pytorch compile issue...
    # TODO: replace this with a proper checkpoint decompilation
    # th._dynamo.config.suppress_errors = True

    split_types = args.split_types
    wandb_model_types = args.wandb_model_types
    local_model_types = args.local_model_types
    batch_cutoff = int(1e6)

    model_type_to_model_cls = {
        "neims": NeimsPL,
        "massformer": MassFormerPL,
        "fragnnet_d3": FraGNNetPL,
        "fragnnet_d4": FraGNNetPL,
        "iceberg_inten": IcebergIntenPL,
        "iceberg_inten_opt": IcebergIntenPL,
        "graff": GrAFFPL,
        "precursor": PrecursorPL,
    }

    model_type_to_ds_cls = {
        "neims": SpecMolDataset,
        "massformer": SpecMolDataset,
        "fragnnet_d3": SpecMolFragDataset,
        "fragnnet_d4": SpecMolFragDataset,
        "iceberg_inten": SpecMolMagmaIntenDataset,
        "iceberg_inten_opt": SpecMolMagmaIntenDataset,
        "graff": SpecMolAnnDataset,
        "precursor": SpecMolDataset,
    }

    model_type_to_wandb_prefix = {
        "neims": "neims_long",
        "massformer": "massformer_long",
        "fragnnet_d3": "fraggnn_d3_long",
        "fragnnet_d4": "fraggnn_d4_long",
        "iceberg_inten": "iceberg_inten3_100_long",
        "iceberg_inten_opt": "iceberg_inten3_all_long",
        "graff": "graff3_long",
        "precursor": "precursor",
    }

    dset_to_wandb_prefix = {
        "nist20": "nist20v3-1",
    }

    model_to_run_id = NestedDefaultDict()
    model_to_config = NestedDefaultDict()

    # find wandb checkpoints
    template_fp = "config/models_template"
    api = wandb.Api()
    for split_type in split_types:
        for model_type in wandb_model_types:
            model_type_wandb_prefix = model_type_to_wandb_prefix[model_type]
            dset_wandb_group_prefix = dset_to_wandb_prefix[dset]
            model_to_config[split_type][model_type] = (
                f"config/{dset}_{split_type}/{model_type}/s0.yml"
            )
            if model_type != "precursor":
                group_name = f"{dset_wandb_group_prefix}_{split_type}_{model_type_wandb_prefix}"
                runs = api.runs(path="frag-gnn/frag-gnn", filters={"group": group_name})
                for run in runs:
                    seed = int(run.name.split("_")[-1].removeprefix("s"))
                    if seed < num_seeds:
                        model_to_run_id[split_type][model_type][seed] = run.id
    del api
    pprint(model_to_config)
    pprint(model_to_run_id)

    model_types = wandb_model_types

    for split_type in split_types:
        for model_type in model_types:
            print(f">>> Starting model_type {model_type}")
            custom_fp = model_to_config[split_type][model_type]

            # Setup configuration
            config_d = load_config(template_fp, custom_fp)
            config_d = apply_inference_config(
                config_d,
                auxiliary_scores=auxiliary_scores,
                skip_extra_losses=skip_extra_losses,
                disable_preproc=disable_preproc,
                output_formula_str=True,
                model_type=model_type,
            )

            print(">>> Loading dataset")
            ds_cls = model_type_to_ds_cls[model_type]
            ds = ds_cls(split=eval_split, **config_d)
            dl = init_dataloader(ds, config_d)
            model_cls = model_type_to_model_cls[model_type]
            device = th.device("cuda:0") if config_d["accelerator"] == "gpu" else th.device("cpu")
            for seed in range(num_seeds):
                print(f">>> Starting seed {seed}")
                if model_type in model_to_run_id[split_type]:
                    assert model_type in wandb_model_types, model_type
                    run_id = model_to_run_id[split_type][model_type][seed]
                    model = init_wandb_ckpt(
                        run_id, last_ckpt=False, model_cls=model_cls, config_d=config_d
                    )
                else:
                    assert model_type == "precursor", model_type
                    # does not need a checkpoint
                    model = model_cls(**config_d)
            model.to(device)
            model.eval()

            # Prepare output subset with model-specific metrics
            output_subset = prepare_output_subset(
                output_everything=output_everything,
                output_subset=set(args.output_subset) if not output_everything else None,
                model_metric_names=model.metric_names,
            )

            # run inference
            vals = run_inference_tabular(
                dl,
                model,
                device,
                eval_split,
                batch_cutoff,
                config_d["nb_iso"],
                output_set=output_subset,
                untransform_spec=untransform_spec,
            )

            # Save results
            save_inference_results(
                vals=vals,
                save_dp=save_dp,
                model_type=model_type,
                seed=seed,
                eval_split=eval_split,
                force_overwrite=force_overwrite,
                split_type=split_type,
            )

            # Cleanup
            cleanup_model(model, use_gpu=(config_d["accelerator"] == "gpu"))
