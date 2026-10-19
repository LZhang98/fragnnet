"""Export RT/CCS debug learning curves stored by SpectrumPL in a trusted checkpoint."""

import argparse
import json
from pathlib import Path

import torch

from fragnnet.models.masster.masster_pl import MassterPL
from fragnnet.utils.migrations import get_pl_hparams


def summarize_checkpoint(checkpoint_path: str) -> dict:
    """Read completed epoch metrics and verify finite trained parameters."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = MassterPL(**get_pl_hparams(checkpoint))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    for name, value in checkpoint["state_dict"].items():
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite checkpoint tensor: {name}")
    names = (
        "loss",
        "primary_loss",
        "rt_loss",
        "ccs_loss",
        "rt_mae",
        "ccs_mae",
        "ccs_mape",
        "rt_label_fraction",
        "ccs_label_fraction",
        "cos_sim_0-01",
    )
    epochs = []
    for epoch in range(checkpoint["epoch"] + 1):
        row = {"epoch": epoch}
        for split in ("train", "val"):
            metrics = checkpoint[f"{split}_mean_metrics"]
            for name in names:
                if name in metrics:
                    value = metrics[name][epoch].item()
                    row[f"{split}_{name}"] = value if torch.isfinite(torch.tensor(value)) else None
        epochs.append(row)
    return {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "global_step": checkpoint["global_step"],
        "parameters_finite": True,
        "strict_reload": True,
        "property_heads": {
            name: type(getattr(model.model, f"{name}_head")).__name__
            for name in model.property_names
        },
        "seed": checkpoint["hyper_parameters"]["seed"],
        "epochs": epochs,
    }


def main() -> None:
    """Write a new JSON report without replacing existing evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = summarize_checkpoint(args.checkpoint)
    with Path(args.output).open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
