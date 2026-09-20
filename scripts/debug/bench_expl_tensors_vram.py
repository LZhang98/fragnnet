"""Benchmark VRAM usage with vs without return_explainability_tensors.

Uses the debug_tinyv4 dataset and a fresh (randomly-initialised) FraGNNet model
so no checkpoint or external data download is required.

Run from repo root:
    conda run -n FRAGNNET-GPU python scripts/debug/bench_expl_tensors_vram.py
"""

import gc
import sys

import torch as th
import yaml

from fragnnet.runner import init_dataloader, init_dataset, load_config
from fragnnet.utils.misc_utils import deep_update
from fragnnet.utils.script_utils import to_device
from fragnnet.utils.setup_utils import get_model_cls

TEMPLATE_FP = "config"
CUSTOM_FP = "config/debug/debug_tiny_gine_gsa.yml"
DEVICE = th.device("cuda" if th.cuda.is_available() else "cpu")
N_BATCHES = 20


def load_debug_data(config_d: dict):
    (ds_val,) = init_dataset(config_d, splits=("val",))
    dl = init_dataloader(ds_val, config_d)
    return dl


def peak_mb(reset: bool = False) -> float:
    if DEVICE.type != "cuda":
        return 0.0
    if reset:
        th.cuda.reset_peak_memory_stats(DEVICE)
    return th.cuda.max_memory_allocated(DEVICE) / 1024**2


def run_forward(model, batch, return_expl: bool) -> None:
    batch_gpu = {
        k: v.to(DEVICE) if isinstance(v, th.Tensor) else v for k, v in batch.items()
    }
    batch_gpu["return_explainability_tensors"] = return_expl
    with th.inference_mode():
        model.model.forward(**{k: v for k, v in batch_gpu.items()
                               if k in model.model.forward.__code__.co_varnames
                               or k == "return_explainability_tensors"
                               or k not in {"batch_size", "spec_unique_id"}})


def bench(model, dl, return_expl: bool, label: str) -> float:
    th.cuda.synchronize(DEVICE)
    gc.collect()
    th.cuda.empty_cache()
    peak_mb(reset=True)

    for i, batch in enumerate(dl):
        if i >= N_BATCHES:
            break
        batch_gpu = to_device(batch, DEVICE)
        batch_gpu["return_explainability_tensors"] = return_expl
        with th.inference_mode():
            _ = model.forward(**batch_gpu)

    th.cuda.synchronize(DEVICE)
    mb = peak_mb()
    print(f"  {label:40s}  peak VRAM = {mb:.1f} MB")
    return mb


def main():
    if DEVICE.type != "cuda":
        print("No CUDA device found — VRAM benchmark requires a GPU.")
        sys.exit(1)

    print(f"Device: {th.cuda.get_device_name(DEVICE)}")
    print(f"Total VRAM: {th.cuda.get_device_properties(DEVICE).total_memory / 1024**2:.0f} MB\n")

    config_d = load_config(TEMPLATE_FP, CUSTOM_FP, override=None)
    # use larger batch size for more realistic VRAM pressure
    config_d = deep_update(config_d, {"eval_batch_size": 8, "num_workers": 0})

    dl = load_debug_data(config_d)

    model_cls = get_model_cls(config_d["model_type"])
    model = model_cls(**config_d).to(DEVICE).eval()

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Batches measured: {N_BATCHES}\n")

    mb_with = bench(model, dl, return_expl=True,  label="return_explainability_tensors=True")
    mb_without = bench(model, dl, return_expl=False, label="return_explainability_tensors=False")

    delta = mb_with - mb_without
    print(f"\n  Delta: {delta:+.1f} MB  ({delta / mb_with * 100:+.1f}%)")


if __name__ == "__main__":
    main()
