"""Loading a model for inference: checkpoint lookup (local path or W&B run), the checkpoint's
config with the inference overlay, model construction and device setup, saving results and
freeing the model (split out of ``utils/script_utils.py``).
"""

import gc
import multiprocessing
import os
import tempfile
from typing import Any

import pandas as pd
import torch as th

from fragnnet import registry
from fragnnet.config import (
    find_unknown_config_keys,
    load_config,
    load_custom_config,
    raise_if_retired_inst_types,
)
from fragnnet.utils.migrations import get_pl_hparams
from fragnnet.utils.misc_utils import deep_update, get_best_ckpt_from_wandb
from fragnnet.utils.nn_utils import decompile_jit_ckpt
from fragnnet.utils.setup_utils import get_model_cls


def init_wandb_ckpt(
    run_id: str,
    last_ckpt: bool,
    model_cls: type,
    config_d: dict[str, Any],
) -> th.nn.Module:
    """Initialize model from W&B checkpoint.

    Args:
        run_id: W&B run ID to download from
        last_ckpt: Whether to load the last checkpoint (True) or best checkpoint (False)
        model_cls: Model class to instantiate
        config_d: Configuration dictionary for model initialization

    Returns:
        Initialized and loaded model
    """
    import wandb

    # load model
    with tempfile.TemporaryDirectory(dir=os.getcwd()) as temp_dp:
        api = wandb.Api()
        run = api.run(f"frag-gnn/frag-gnn/{run_id}")
        if last_ckpt:
            ckpt_wandb_fp = "ckpt/last.ckpt"
        else:
            files = run.files(per_page=100)
            best_ckpt_wandb_fp, best_epoch = None, None
            for file in files:
                if file.name.endswith(".ckpt") and "last" not in file.name:
                    ckpt_wandb_fp = file.name
                    epoch = int(
                        os.path.basename(file.name)
                        .removeprefix("model-epoch=")
                        .removesuffix(".ckpt")
                    )
                    if best_ckpt_wandb_fp is None or epoch > best_epoch:
                        best_ckpt_wandb_fp = ckpt_wandb_fp
                        best_epoch = epoch
            ckpt_wandb_fp = best_ckpt_wandb_fp  # bugfix: assign winning path after loop
        assert ckpt_wandb_fp is not None
        ckpt_file_local = run.file(ckpt_wandb_fp).download(root=temp_dp, replace=False)
        ckpt_fp = ckpt_file_local.name
        ckpt = th.load(ckpt_fp, map_location="cpu", weights_only=False)
        # decompile
        assert not config_d["compile"]
        ckpt = decompile_jit_ckpt(ckpt)
        # init
        model = model_cls(**config_d)
        model.load_state_dict(ckpt["state_dict"], strict=True)
    return model


def setup_device(
    config_d: dict[str, Any],
    device: str | None = None,
    enable_cpu_optimizations: bool = True,
) -> th.device:
    """
    Prepare and configure the compute device for inference.

    Args:
            config_d: Configuration dictionary (should contain 'accelerator' key)
            device: Device string (e.g., "cpu", "cuda:0"). Auto-detects if None.
            enable_cpu_optimizations: Whether to enable CPU-specific optimizations

    Returns:
            Configured torch device
    """
    if device is None:
        if config_d.get("accelerator") == "gpu" and th.cuda.is_available():
            n_devices = th.cuda.device_count()
            if n_devices != 1:
                print(
                    f"Warning: {n_devices} CUDA devices visible; using cuda. "
                    "Set CUDA_VISIBLE_DEVICES to a single GPU or MIG UUID to be explicit."
                )
            th_device = th.device("cuda")
        else:
            th_device = th.device("cpu")
    elif not th.cuda.is_available() and device != "cpu":
        raise RuntimeError(f"Requested device '{device}' but CUDA is not available.")
    else:
        th_device = th.device(device)

    print(f"> Using device: {th_device}")

    # CPU-specific optimizations
    if th_device.type == "cpu" and enable_cpu_optimizations:
        th.set_num_threads(multiprocessing.cpu_count())
        th.multiprocessing.set_sharing_strategy("file_system")
        th.use_deterministic_algorithms(True)
        print(f"  - CPU threads: {multiprocessing.cpu_count()}")
        print("  - Deterministic mode: enabled")

    return th_device


def get_ckpt_path(
    wandb_run_id: str | None,
    model_ckpt: str | None,
    model_save_dp: str,
    use_cached: bool = True,
) -> str:
    """
    Get checkpoint path either from local file or download from W&B.

    Args:
            wandb_run_id: W&B run ID to download from
            model_ckpt: Local checkpoint path
            model_save_dp: Directory to save downloaded checkpoints
            use_cached: Whether to use cached W&B downloads

    Returns:
            Path to checkpoint file
    """
    if model_ckpt is not None:
        if not os.path.isfile(model_ckpt):
            raise FileNotFoundError(f"Local checkpoint does not exist: {model_ckpt}")
        ckpt_fp = model_ckpt
        print(f">> Using local checkpoint: {ckpt_fp}")
    else:
        if wandb_run_id is None:
            raise ValueError("Either wandb_run_id or model_ckpt must be provided")
        os.makedirs(model_save_dp, exist_ok=True)
        print(f">> Downloading model to {model_save_dp} from W&B")
        print(f">> W&B run id: {wandb_run_id}")
        ckpt_fp = get_best_ckpt_from_wandb(model_save_dp, wandb_run_id, use_cached=use_cached)

    return ckpt_fp


def init_model_from_ckpt(
    ckpt_fp: str,
    config_d: dict[str, Any],
    device: th.device,
    map_location: str | None = None,
) -> th.nn.Module:
    """
    Initialize and load model from checkpoint file.
    Args:
            ckpt_fp: Path to checkpoint file
            config_d: Configuration dictionary
            device: Device to load model on
            map_location: Location to map checkpoint tensors (defaults to device)

    Returns:
            Loaded model
    """
    if map_location is None:
        map_location = str(device)

    # Load checkpoint
    print(f">> Loading checkpoint from: {ckpt_fp}")
    ckpt = th.load(ckpt_fp, map_location=map_location, weights_only=False)

    # Use saved compilation metadata: inference setup already disables compilation
    # in config_d, but compiled checkpoints still contain _orig_mod state keys.
    ckpt = decompile_jit_ckpt(ckpt)
    config_d["compile"] = False
    # Get model class and initialize
    model_type = config_d["model_type"]
    print(f">> Model type: {model_type}")
    model_cls = get_model_cls(model_type)

    model = model_cls(**config_d)

    # Inference requires complete trained weights; partial loads can leave random layers.
    model.load_state_dict(ckpt["state_dict"], strict=True)
    print(">> Model loaded successfully (strict=True)")

    model.to(device)
    model.eval()

    return model


def get_config_from_ckpt(
    ckpt_fp: str,
    custom_fp: str | None = None,
    template_fp: str | None = None,
) -> dict[str, Any]:
    """
    Load model and config from checkpoint file (using checkpoint hparams).

    This is the recommended approach for MS2C prediction scripts or when you
    want to use the exact configuration that was used during training.

    Args:
            ckpt_fp: Path to checkpoint file
            custom_fp: YAML overlay. Only explicitly declared settings override checkpoint
                values; nested mappings merge without resetting their other saved fields.
            template_fp: Template used to validate the overlay, required with custom_fp.
                Its defaults do not replace checkpoint settings. Missing legacy settings
                are supplied only by checkpoint migrations.

    Returns:
            Configuration dictionary
    """
    # Load checkpoint to extract config
    print(">> Loading checkpoint to extract config")
    ckpt = th.load(ckpt_fp, map_location="cpu", weights_only=False)

    print(">> Extracting config from checkpoint hparams")
    config_d = get_pl_hparams(ckpt)

    # Override with custom config if provided
    if custom_fp is not None and template_fp is None:
        raise ValueError("template_fp required when custom_fp is provided")
    # Load and merge template and custom config
    if template_fp is not None and custom_fp is not None:
        print(f">> Loading custom config from {custom_fp}")
        # The checkpoint's own model_type picks the overlay when the custom config is silent,
        # so an inference config never pulls in another model's keys.
        schema = load_config(
            template_fp,
            None,
            override=None,
            model_type=config_d.get("model_type"),
        )
        custom_d = load_custom_config(custom_fp)
        unknown = find_unknown_config_keys(custom_d, schema)
        if unknown:
            raise ValueError(f"{custom_fp} sets keys the template does not declare: {unknown}")
        overlay_model_type = custom_d.get("model_type", schema["model_type"])
        registry.get_model_hooks(overlay_model_type)
        if registry.MODEL_REGISTRY[overlay_model_type] != registry.MODEL_REGISTRY[schema["model_type"]]:
            raise ValueError("inference overlay cannot change the checkpoint's model_type")
        config_d = deep_update(config_d, custom_d)
        raise_if_retired_inst_types(config_d, "inference config")
    return config_d


def apply_inference_config(
    config_d: dict[str, Any],
    auxiliary_scores: list | None = None,
    skip_extra_losses: bool | None = None,
    disable_preproc: bool | None = False,
    eval_mz_bin_res: list | None = None,
    dynamic_batch_sampler: bool | None = None,
    num_workers: int | None = None,
    output_formula_str: bool = False,
    train_batch_size: int | None = None,
    eval_batch_size: int | None = None,
    drop_empty_spectra: bool | None = None,
    # Set dataframes in config for dataset initialization
    mol_src: pd.DataFrame | str | None = None,
    spec_src: pd.DataFrame | str | None = None,
) -> dict[str, Any]:
    """Apply inference-specific configuration modifications.

    Modifies configuration dictionary for inference evaluation, including batch sizes,
    preprocessing settings, output options, and model-specific parameters.

    Args:
        config_d: Configuration dictionary to modify
        auxiliary_scores: List of auxiliary score types to compute
        skip_extra_losses: Whether to skip extra loss terms
        disable_preproc: Whether to disable preprocessing in all modules
        eval_mz_bin_res: M/z bin resolutions for evaluation scoring
        dynamic_batch_sampler: Whether to use dynamic batch sampling
        num_workers: Number of data loading workers
        output_formula_str: Whether to output formula strings
        train_batch_size: Training batch size override
        eval_batch_size: Evaluation batch size override
        drop_empty_spectra: Override for spec_params.drop_empty_spectra. Pass False for
            prediction-only runs, whose spec_df carries no ground-truth peaks. None leaves
            the checkpoint's setting alone (evaluation runs score against real spectra).
        mol_src: DataFrame or path to molecule source data
        spec_src: DataFrame or path to spectrum source data

    Returns:
        Modified configuration dictionary
    """
    # Common configuration modifications
    if auxiliary_scores is not None:
        config_d["auxiliary_scores"] = auxiliary_scores
    if skip_extra_losses is not None:
        config_d["skip_extra_losses"] = skip_extra_losses
    config_d["compile"] = False  # Disable compilation for evaluation
    # Disable preprocessing if requested
    if disable_preproc:
        for k in ["spec", "mol"]:
            key = f"{k}_params"
            if key not in config_d:
                continue
            config_d[key]["preprocess"] = False

    # Handle formula string outputs
    config_d["output_formula_str"] = output_formula_str

    # The formula-delta export reads spec_prec_type_str for every model.
    config_d["spec_params"]["prec_type_str"] = True
    # The model's own blocks (FraGNNet's frag_params, GRAFF's ann_params).
    registry.get_model_hooks(config_d.get("model_type"), strict=False).apply_inference_config(
        config_d, disable_preproc=bool(disable_preproc), output_formula_str=output_formula_str
    )
    # Set batch sizes if provided
    if train_batch_size is not None:
        config_d["train_batch_size"] = train_batch_size
    if eval_batch_size is not None:
        config_d["eval_batch_size"] = eval_batch_size

    # config_d['auxiliary_scores'] = auxiliary_scores if auxiliary_scores is not None else []
    if eval_mz_bin_res is not None:
        config_d["eval_mz_bin_res"] = eval_mz_bin_res

    # Ground-truth peak cleanup is meaningless without ground-truth peaks: an inference
    # spec_df has `peaks == []` by construction, so drop_empty_spectra (True in
    # config/template.yml) removes every spectrum and the run silently predicts nothing.
    # Only override when the checkpoint config carries the cleanup group - writing the key
    # into a legacy config that omits it would trip the missing-cleanup-params check in
    # BaseDataset._setup_dfs.
    if drop_empty_spectra is not None and "drop_empty_spectra" in config_d["spec_params"]:
        config_d["spec_params"]["drop_empty_spectra"] = drop_empty_spectra

    # Apply common inference settings
    config_d["track_datapoint_metrics"] = False

    # Disable samplers for deterministic inference
    config_d["dynamic_batch_sampler"] = (
        dynamic_batch_sampler if dynamic_batch_sampler is not None else False
    )
    config_d["group_sampler"] = False
    config_d["simple_group_sampler"] = False

    if num_workers is not None:
        config_d["num_workers"] = num_workers

    # Set dataframes in config for dataset initialization
    if mol_src is not None:
        config_d["mol_fp"] = mol_src
    if spec_src is not None:
        config_d["spec_fp"] = spec_src
    return config_d


def save_inference_results(
    vals: Any,
    save_dp: str,
    model_type: str,
    seed: int,
    eval_split: str,
    force_overwrite: bool = False,
    split_type: str | None = None,
) -> None:
    """
    Save inference results to disk with proper directory structure.

    Args:
            vals: Results to save (typically a dictionary or DataFrame)
            save_dp: Base directory to save results
            model_type: Model type identifier
            seed: Random seed number
            eval_split: Dataset split name
            force_overwrite: Whether to overwrite existing files
            split_type: Optional split type (e.g., "inchikey", "scaffold")
    """
    # Build save path
    if split_type is not None:
        seed_save_dp = os.path.join(save_dp, split_type, model_type, f"s{seed}")
    else:
        seed_save_dp = os.path.join(save_dp, model_type, f"s{seed}")

    os.makedirs(seed_save_dp, exist_ok=True)
    seed_vals_fp = os.path.join(seed_save_dp, f"{eval_split}.pkl")

    # Save with overwrite logic
    if os.path.isfile(seed_vals_fp):
        print(f"> Info: {seed_vals_fp} exists already")
        if force_overwrite:
            print(f"> Warning: overwriting {seed_vals_fp}")
            th.save(vals, seed_vals_fp)
        else:
            print("> Skipping save (use --force_overwrite to overwrite)")
    else:
        th.save(vals, seed_vals_fp)
        print(f"> Saved results to {seed_vals_fp}")


def cleanup_model(model: th.nn.Module, use_gpu: bool = True) -> None:
    """
    Clean up model and free memory. This is useful after inference crashed to avoid memory leaks.

    Args:
            model: Model to clean up
            use_gpu: Whether GPU memory should be cleared
    """
    model.cpu()
    del model
    gc.collect()

    if use_gpu:
        with th.no_grad():
            th.cuda.empty_cache()
