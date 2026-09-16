import argparse
import os
import sys
import tempfile
import time
from typing import cast

import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from fragnnet.dataset import IterableSpecMolFragDataset, SpecMolFragDataset
from fragnnet.runner import init_dataloader
from fragnnet.utils import data_utils, frag_utils
from fragnnet.utils.misc_utils import booltype
from fragnnet.utils.script_utils import (
    apply_inference_config,
    available_frag_mol_ids,
    get_ckpt_path,
    get_config_from_ckpt,
    get_required_config_indices,
    get_tabular_columns,
    init_model_from_ckpt,
    list_output_subset_options,
    prepare_output_subset,
    run_inference_tabular_iter,
    setup_device,
)
from fragnnet.utils.setup_utils import get_dataset_cls


def passthrough_collate(batch):
    """Identity collate: IterableSpecMolFragDataset already yields collated batches.

    Defined at module level rather than as a lambda so DataLoader workers can pickle it.
    """
    return batch


if __name__ == "__main__":
    # Early check for --list_output_options to avoid required argument validation
    if "--list_output_options" in sys.argv:
        print("Available output_subset options:")
        for opt in list_output_subset_options():
            print(f"  - {opt}")
        sys.exit(0)

    # Parse command line arguments
    parser = argparse.ArgumentParser()

    # Model configuration
    parser.add_argument(
        "--template_fp",
        type=str,
        default="config/template.yml",
        help="Path to the template config file",
    )
    parser.add_argument("--custom_fp", type=str, default=None, help="Path to custom config file")
    # Model checkpoint loading
    parser.add_argument(
        "--wandb_run_id",
        type=str,
        default=None,
        help="W&B run ID to download checkpoint from (if model_ckpt not provided)",
    )
    parser.add_argument(
        "--model_save_dp",
        type=str,
        default="saved_ckpts",
        help="Directory to save/cache downloaded W&B checkpoints",
    )
    parser.add_argument(
        "--model_ckpt",
        type=str,
        default=None,
        help="Path to local model checkpoint (if None, will download from W&B using wandb_run_id)",
    )

    # Output configuration
    parser.add_argument(
        "--save_dp",
        type=str,
        default="output/inference_outputs",
        help="Directory to save inference evaluation results",
    )
    parser.add_argument(
        "--save_format",
        type=str,
        default="pkl",
        choices=["pkl", "pkl.gz"],
        help="File format for saving predictions. Parquet is not supported: prediction "
        "cells are torch.Tensors, which pyarrow cannot serialize.",
    )
    # Output control arguments
    parser.add_argument(
        "--list_output_options",
        action="store_true",
        help="List all valid tokens for --output_subset and exit",
    )
    parser.add_argument(
        "--output_subset",
        type=str,
        nargs="+",
        default=[
            "pred_mzs",  # Predicted m/z values
            "pred_ints",  # Predicted intensities
        ],
        # "pred_mzs",
        # "pred_ints",
        # "pred_oos_prob",
        # "frag_formula_peak_mzs",
        # "pred_formula_probs",
        # "pred_formula_node_probs",
        # "pred_node_formula_probs",
        # "pred_node_probs"
        help="Subset of outputs to include in results",
    )
    parser.add_argument(
        "--output_everything",
        type=booltype,
        default=False,
        help="Whether to output all columns (overrides output_subset)",
    )
    parser.add_argument(
        "--return_formula_str",
        type=booltype,
        default=True,
        help="Whether to return formula strings",
    )
    parser.add_argument(
        "--return_cc_hash", type=booltype, default=True, help="Whether to return node CC hashes"
    )
    parser.add_argument(
        "--drop_none_cols",
        type=booltype,
        default=True,
        help="Drop columns that are entirely None/NaN before saving outputs",
    )
    # Inference configuration
    parser.add_argument(
        "--skip_extra_losses",
        type=booltype,
        default=False,
        help="Skip computing auxiliary losses during inference for speed",
    )
    parser.add_argument(
        "--disable_preproc",
        type=booltype,
        default=True,
        help="Disable dataset preprocessing (set to True for inference to save memory)",
    )

    # Data paths
    parser.add_argument("--mol_fp", type=str, required=True, help="file path to mol_df.pkl.gz")
    parser.add_argument("--spec_fp", type=str, required=True, help="file path to spec_df.pkl.gz")

    parser.add_argument(
        "--frag_dp",
        type=str,
        required=False,
        default=None,
        help="Directory containing precomputed fragment graphs; if None, will auto-generate in temp directory",
    )

    # Prediction filtering
    parser.add_argument(
        "--min_intensity",
        type=float,
        default=1.0,
        help="Minimum intensity threshold for peaks on a 0-1000 scale (e.g. 1.0 = 0.1%% of base peak); filters out peaks at or below this value",
    )

    # Performance tuning
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Fixed batch size for evaluation/inference; ignored when --dynamic_batch_max is set. "
        "Matches eval_batch_size in config/template.yml. Note that 1 is not just slow: it "
        "also feeds return_batch_at in init_dataloader, so a config with "
        "dynamic_batch_sampler=True would flush after every sample and never reach "
        "dynamic_batch_sampler_max",
    )
    parser.add_argument(
        "--dynamic_batch_max",
        type=int,
        default=None,
        help="If set, pack batches greedily up to this many frag_node/frag_edge "
        "(see --dynamic_batch_limited_by) instead of using a fixed --batch_size. "
        "Uses IterableSpecMolFragDataset, which batches lazily as it streams. "
        "In frag_node units, config/template.yml uses 100000 for training.",
    )
    parser.add_argument(
        "--dynamic_batch_limited_by",
        type=str,
        default="frag_node",
        choices=["frag_node", "frag_edge"],
        help="Quantity capped by --dynamic_batch_max. Only use frag_edge for configs "
        "that materialize fragment edges; otherwise every sample measures 0.",
    )
    parser.add_argument(
        "--save_chunk_size",
        type=int,
        default=10_000,
        help="Number of predictions per chunk file (for resumable inference)",
    )
    # Sharding (SLURM array jobs)
    parser.add_argument(
        "--start_mol_idx",
        type=int,
        default=0,
        help="Row offset into mol_df: this run only predicts molecules "
        "[start_mol_idx, start_mol_idx + num_mols). Use the same offset/size as the "
        "DAG preprocessing array so each task lines up with one frag shard.",
    )
    parser.add_argument(
        "--num_mols",
        type=int,
        default=-1,
        help="Number of mol_df rows in this shard (-1 = all remaining rows).",
    )
    args = parser.parse_args()

    # Get checkpoint path (download from W&B or use local)
    ckpt_fp = get_ckpt_path(
        wandb_run_id=args.wandb_run_id,
        model_ckpt=args.model_ckpt,
        model_save_dp=args.model_save_dp,
        use_cached=True,
    )
    save_dp = args.save_dp
    template_fp = args.template_fp
    custom_fp = args.custom_fp

    # Load model and config from checkpoint
    # This extracts config from checkpoint hparams and applies common inference settings
    print(">>> Loading model and configuration from checkpoint")
    config_d = get_config_from_ckpt(
        ckpt_fp=ckpt_fp,
        custom_fp=custom_fp,
        template_fp=template_fp,
    )

    # Load Input Data (molecules and spectra)
    print("> Loading mol_df")
    mol_df = pd.read_pickle(args.mol_fp)
    print(f"  num mol: {len(mol_df)}")
    print("> Loading spec_df")
    spec_df = pd.read_pickle(args.spec_fp)
    print(f"  num spec: {len(spec_df)}")

    print(mol_df.head(5))
    print(spec_df.head(5))

    # =========================================
    # Shard selection (SLURM array jobs)
    # =========================================
    # Slice by mol_df ROW POSITION, matching how the DAG preprocessing array sharded,
    # so shard i of the frag output holds exactly the molecules selected here.
    if args.start_mol_idx > 0 or args.num_mols >= 0:
        if args.start_mol_idx >= len(mol_df):
            print(
                f"> start_mol_idx {args.start_mol_idx} is beyond mol_df ({len(mol_df)} rows), "
                "nothing to do"
            )
            sys.exit(0)
        stop = len(mol_df) if args.num_mols < 0 else args.start_mol_idx + args.num_mols
        mol_df = mol_df.iloc[args.start_mol_idx : stop]
        spec_df = spec_df[spec_df["mol_id"].isin(set(mol_df["mol_id"]))]
        print(
            f"> Shard [{args.start_mol_idx}, {args.start_mol_idx + len(mol_df)}): "
            f"{len(mol_df)} molecules, {len(spec_df)} spectra"
        )
        if len(spec_df) == 0:
            print("> Shard has no spectra, nothing to do")
            sys.exit(0)

    # =========================================
    # Resume logic
    # =========================================
    start_chunk_idx = 0

    os.makedirs(args.save_dp, exist_ok=True)
    print(f"> Checking for existing predictions in {args.save_dp}...")
    existing_files = [
        f
        for f in os.listdir(args.save_dp)
        if f.startswith("predictions_chunk_") and f.endswith(args.save_format)
    ]

    finished_spec_ids = set()
    max_chunk_idx = -1

    for f in tqdm(existing_files, desc="Reading existing predictions"):
        # Parse chunk index
        try:
            c_idx = int(f.split("_")[-1].split(".")[0])
            max_chunk_idx = max(max_chunk_idx, c_idx)
        except Exception:
            pass

        fp = os.path.join(args.save_dp, f)
        try:
            df = pd.read_pickle(fp)

            if "spec_id" in df.columns:
                finished_spec_ids.update(df["spec_id"].unique())
        except Exception as e:
            print(f"Warning: Could not read {fp}: {e}")

    # Filter once, after every chunk has been read. Doing it inside the loop re-filtered
    # spec_df/mol_df once per chunk file and made the "skipped" count report only the
    # spectra the latest file contributed.
    if existing_files:
        print(f"> Found {len(finished_spec_ids)} finished spectra.")
        start_chunk_idx = max_chunk_idx + 1

        # Filter spec_df
        original_len = len(spec_df)
        spec_df = spec_df[~spec_df["spec_id"].isin(finished_spec_ids)]
        print(
            f"> Resuming inference on {len(spec_df)} spectra (skipped {original_len - len(spec_df)})"
        )

        # Sync mol_df
        valid_mol_ids = spec_df["mol_id"].unique()
        mol_df = mol_df[mol_df["mol_id"].isin(valid_mol_ids)]
        print(f"> Filtered mol_df to {len(mol_df)} molecules based on remaining spectra")

        if len(spec_df) == 0:
            print("All spectra processed. Exiting.")
            exit(0)

    # =========================================
    # Repare for Inference
    # =========================================

    # Apply inference-specific configuration adjustments
    apply_inference_config(
        config_d,
        skip_extra_losses=args.skip_extra_losses,
        disable_preproc=args.disable_preproc,
        eval_batch_size=args.batch_size,
        auxiliary_scores=[],
        # Prediction-only input: spec_df has no ground-truth peaks to clean or to drop on.
        drop_empty_spectra=False,
    )

    # Fragment graphs (DAGs) are computationally expensive to generate
    # Options: 1) Use pre-computed fragments, 2) Generate in temporary directory
    # Preserve the CLI's legacy FraGNNet spelling while using canonical dispatch.
    ds_cls = get_dataset_cls(
        "frag_gnn" if config_d["model_type"] == "fragnnet" else config_d["model_type"]
    )
    requires_dags = issubclass(ds_cls, SpecMolFragDataset)
    temp_frag_dir = None
    if args.frag_dp is not None:
        # Use pre-computed fragments
        config_d["frag_dp"] = args.frag_dp
        print(f"> Using provided fragment directory: {config_d['frag_dp']}")
    else:
        # Auto-generate fragments in temporary directory (only for FraGNNet models)
        temp_frag_dir = tempfile.TemporaryDirectory() if requires_dags else None
        if requires_dags:
            if temp_frag_dir is None:
                raise RuntimeError("Temporary fragment directory could not be created")
            print(f"> Using temporary fragment directory: {temp_frag_dir.name}")
            config_d["frag_dp"] = temp_frag_dir.name

            # Prepare inputs for parallel fragment graph generation
            dag_feat_inputs_l = []
            for _, row in tqdm(
                mol_df.iterrows(), total=len(mol_df), desc="Preparing fragment inputs"
            ):
                dag_feat_inputs_l.append(
                    [
                        row["mol"],  # RDKit molecule object
                        row["mol_id"],  # Molecule identifier
                        config_d["num_depth"] - 1,  # Maximum fragmentation depth
                        True,  # Use hydrogen prior
                        config_d["num_hs"],  # Maximum hydrogen transfers
                        150,  # Maximum time for fragmentation (seconds)
                        False,  # Include isotopes
                        False,  # Collapse isomorphic fragments
                        -1,  # WL hashing max iterations (-1 disables hash computation)
                        False,  # Disable multi-cut BFS
                        2,  # Maximum cut size if multi-cut BFS is enabled
                        False,  # Disable SMARTS prepass
                    ]
                )

            # Time the fragment generation process
            start = time.time()
            print(
                f"> Running parallel fragment generation for {len(dag_feat_inputs_l)} molecules..."
            )
            # Generate fragment graphs in parallel across CPU cores
            frag_results = data_utils.par_apply(
                iter(dag_feat_inputs_l), frag_utils.timed_get_dags, True, return_as_generator=True
            )
            if frag_results is None:
                raise RuntimeError("Parallel fragment generation did not return any results")
            frag_results_iter = cast("list[tuple[str, dict]]", frag_results)
            # Consume the generator to complete all fragment computations
            failed_frag_count = 0
            for mol_id, dag_d in tqdm(
                frag_results_iter, total=len(dag_feat_inputs_l), desc="Computing Fragments"
            ):
                if not dag_d:
                    failed_frag_count += 1
                    continue
                frag_utils.save_frag_d(
                    frag_d=dag_d,
                    mol_id=mol_id,
                    frag_dp=config_d["frag_dp"],
                    is_compressed=config_d["frag_params"]["compressed"],
                )
            end = time.time()
            time_cost = end - start
            print(
                f"> Fragment generation completed in {time_cost:.2f}s ({len(dag_feat_inputs_l) / time_cost:.2f} mol/s)"
            )
            print(f"> Fragment generation failures: {failed_frag_count}")

    # Filter input data to only include molecules with successfully generated fragments
    if requires_dags:
        print(">> Filtering molecules without fragments...")
        all_frags = available_frag_mol_ids(config_d["frag_dp"])
        print(f">> Found {len(all_frags)} fragment entries")
        mol_df = mol_df[mol_df["mol_id"].astype(str).isin(all_frags)]
        print(f"> Molecules after filtering: {len(mol_df)}")
        spec_df = spec_df[spec_df["mol_id"].isin(mol_df["mol_id"].to_list())]
        print(f"> Spectra after filtering: {len(spec_df)}")
        # Report this cause here rather than letting the empty-dataset guard below blame
        # the dataset: nothing reaches it, so its message about spec_params would send the
        # reader looking in the wrong place.
        if len(spec_df) == 0:
            raise RuntimeError(
                f"No molecule has a fragment graph in {config_d['frag_dp']} "
                f"({len(all_frags)} entries found), so every spectrum was filtered out. "
                "Check --frag_dp, or drop it to generate fragments on the fly."
            )

    # Set dataframes in config for dataset initialization
    config_d["mol_fp"] = mol_df
    config_d["spec_fp"] = spec_df

    # Initialize Dataset and DataLoader
    print(">>> Initializing dataset...")
    ds = ds_cls(split="predict_only", **config_d)
    print(f">>> Dataset size: {len(ds)} spectra")
    # The legitimate "nothing to do" cases (empty shard, fully resumed) already exited
    # above, so an empty dataset here means the dataset silently dropped every spectrum
    # it was handed. Fail instead of writing an empty prediction set and exiting 0.
    if len(ds) == 0:
        raise RuntimeError(
            f"Dataset is empty although {len(spec_df)} spectra reached it; the dataset "
            "dropped all of them. Check the checkpoint's spec_params against the input: "
            "prec_types, inst_types, split_filters and drop_empty_spectra."
        )
    if args.dynamic_batch_max is not None:
        stream_ds = IterableSpecMolFragDataset(
            dataset=ds,
            index_stream=range(len(ds)),
            max_num=args.dynamic_batch_max,
            limited_by=args.dynamic_batch_limited_by,
        )
        dl = DataLoader(
            stream_ds,
            # The dataset yields collated batches, so disable a second round of batching.
            # Keep iteration in this process: worker-based iteration of this nested dataset
            # currently deadlocks before yielding its first batch.
            batch_size=None,
            collate_fn=passthrough_collate,
            num_workers=0,
            pin_memory=config_d["pin_memory"] and (config_d["accelerator"] != "cpu"),
        )
        print(
            f">>> Dynamic batching: max_{args.dynamic_batch_limited_by}="
            f"{args.dynamic_batch_max} (batch count known only once the stream is consumed)"
        )
    else:
        dl = init_dataloader(ds, config_d)

    device = setup_device(config_d)
    model = init_model_from_ckpt(ckpt_fp, config_d, device)
    print(f">>> Model loaded and ready on device: {device}")
    print(f">>> Model in eval mode: {not model.training}")

    # Run inference with periodic saving
    print(">>> Running inference and saving predictions...")
    output_set = prepare_output_subset(
        output_subset=args.output_subset, output_everything=args.output_everything
    )
    # Show which index variables will be required for the chosen output_subset
    if output_set is not None:
        req_batch, req_node, req_formula, req_nb_node = get_required_config_indices(
            output_set,
            model,
            nb_iso=False,
            return_formula_str=args.return_formula_str or args.output_everything,
            return_cc_hash=args.return_cc_hash or args.output_everything,
        )
        print("> Required index variables for output_subset:")
        if req_batch:
            print(f"  Batch indices: {req_batch}")
        if req_node:
            print(f"  Node indices: {req_node}")
        if req_formula:
            print(f"  Formula indices: {req_formula}")

    columns_name = get_tabular_columns(
        config_d,
        return_formula_str=args.return_formula_str or args.output_everything,
        return_cc_hash=args.return_cc_hash or args.output_everything,
        output_subset=output_set,
    )

    # =========================================
    # Inferencing
    # =========================================

    # Iterate through batches and save periodically
    preds_l = []
    chunk_idx = start_chunk_idx
    total_preds = 0
    file_ext = args.save_format

    print(f"> Saving every ~{args.save_chunk_size} predictions")

    for batch_rows in run_inference_tabular_iter(
        model,
        dl,
        device,
        evaluate=False,
        batch_cutoff=-1,
        min_peak_int=args.min_intensity,
        return_formula_str=args.return_formula_str or args.output_everything,
        return_cc_hash=args.return_cc_hash or args.output_everything,
        rebase_indices=True,
        output_set=output_set,
    ):
        # batch_rows is COLUMN-major: one list per column, each holding a value per
        # spectrum in the batch. Transpose to per-spectrum rows for the DataFrame.
        rows = list(zip(*batch_rows))
        preds_l.extend(rows)
        total_preds += len(rows)

        # Save periodically once enough predictions have accumulated (batches may be
        # variable-sized under dynamic batching, so trigger on prediction count, not batch count)
        if len(preds_l) >= args.save_chunk_size:
            pred_df = pd.DataFrame(preds_l, columns=columns_name)
            if args.drop_none_cols:
                before_cols = set(pred_df.columns)
                pred_df = pred_df.dropna(axis=1, how="all")
                dropped = sorted(before_cols - set(pred_df.columns))
                if dropped:
                    print(f"> Dropped all-NaN columns: {dropped}")
            chunk_fp = os.path.join(args.save_dp, f"predictions_chunk_{chunk_idx}.{file_ext}")
            pred_df.to_pickle(chunk_fp)
            print(f">> Saved {len(preds_l)} predictions to {chunk_fp} (total: {total_preds})")
            preds_l = []
            chunk_idx += 1

    # Save remaining predictions
    if len(preds_l) > 0:
        pred_df = pd.DataFrame(preds_l, columns=columns_name)
        if args.drop_none_cols:
            before_cols = set(pred_df.columns)
            pred_df = pred_df.dropna(axis=1, how="all")
            dropped = sorted(before_cols - set(pred_df.columns))
            if dropped:
                print(f"> Dropped all-NaN columns: {dropped}")
        chunk_fp = os.path.join(args.save_dp, f"predictions_chunk_{chunk_idx}.{file_ext}")
        pred_df.to_pickle(chunk_fp)
        print(f">> Saved {len(preds_l)} predictions to {chunk_fp} (total: {total_preds})")

    print(f"\n>>> Inference complete! Total predictions: {total_preds}")
    print(f">>> Predictions saved to: {args.save_dp}")

    # =========================================
    # Clean up temporary fragment directory
    # =========================================
    if temp_frag_dir is not None:
        print(f">>> Cleaning up temporary fragment directory: {temp_frag_dir.name}")
        temp_frag_dir.cleanup()
