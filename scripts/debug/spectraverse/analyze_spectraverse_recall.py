"""Rescore Spectraverse MASSter validation recall and candidate coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch as th
from torch.utils.data import DataLoader

from fragnnet.inference.checkpoint import init_model_from_ckpt
from fragnnet.runner import load_config
from fragnnet.utils.setup_utils import get_dataset_cls


def _coverage(target_mzs, target_ints, predicted_mzs, rel, min_mz):
    """Return target-peak and target-intensity recall at a relative mass tolerance."""
    predicted_mzs = np.sort(np.asarray(predicted_mzs, dtype=np.float64))
    target_mzs = np.asarray(target_mzs, dtype=np.float64)
    target_ints = np.asarray(target_ints, dtype=np.float64)
    if not len(predicted_mzs):
        return 0.0, 0.0
    right = np.searchsorted(predicted_mzs, target_mzs)
    left = np.maximum(right - 1, 0)
    right = np.minimum(right, len(predicted_mzs) - 1)
    distance = np.minimum(
        np.abs(predicted_mzs[left] - target_mzs),
        np.abs(predicted_mzs[right] - target_mzs),
    )
    matched = distance <= rel * np.maximum(target_mzs, min_mz)
    return float(matched.mean()), float(target_ints[matched].sum() / target_ints.sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config("config/template.yml", f"config/spectraverse/masster/cv{args.fold}.yml", [])
    config["num_workers"] = 0
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    if device.type == "cpu":
        config["masster_candidate_backend"] = "cpu"
    source = pd.read_pickle(config["spec_fp"])
    ids = pd.read_csv(Path(config["split_dp"]) / "val_ids.csv")["spec_id"]
    selected = source[source.spec_id.isin(ids)].sample(args.sample_size, random_state=20261001)
    selected = selected.sort_values("spec_id")
    molecules = pd.read_pickle(config["mol_fp"])
    subset = args.output.parent / f"spv_cv{args.fold}_subset"
    subset.mkdir(parents=True, exist_ok=True)
    selected.to_pickle(subset / "spec_df.pkl")
    molecules[molecules.mol_id.isin(selected.mol_id)].to_pickle(subset / "mol_df.pkl")
    config["spec_fp"] = str(subset / "spec_df.pkl")
    config["mol_fp"] = str(subset / "mol_df.pkl")
    model = init_model_from_ckpt(str(args.checkpoint), config, device)
    dataset = get_dataset_cls(config["model_type"])(split="val", **config)
    rows = []
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=dataset.get_collate_fn())
    with th.inference_mode():
        for batch_number, batch in enumerate(loader):
            batch = model.transfer_batch_to_device(batch, device, dataloader_idx=0)
            output = model.inference_step(batch, metrics=None)
            true_mzs, true_ints, true_idxs = model.preproc_spec(
                batch["spec_mzs"], batch["spec_ints"], batch["spec_batch_idxs"],
                train=False, pred=False, batch_size=int(batch["batch_size"]),
            )
            arrays = {
                key: output[key].detach().cpu().numpy()
                for key in (
                    "pred_mzs", "pred_batch_idxs", "candidate_mzs_by_cut",
                    "candidate_batch_idxs_by_cut",
                )
            }
            true = [x.detach().cpu().numpy() for x in (true_mzs, true_ints, true_idxs)]
            for index in range(int(batch["batch_size"])):
                spec = dataset.spec_df.iloc[batch_number * 4 + index]
                inst = spec.inst_type
                tol = config["inst_type_loss_tol"][inst]
                target_mask = true[2] == index
                target_mzs, target_ints = true[0][target_mask], true[1][target_mask]
                row = {"spec_id": int(spec.spec_id), "inst_type": inst,
                       "n_target_peaks": len(target_mzs)}
                for kind, mz_key, idx_key in (
                    ("pred", "pred_mzs", "pred_batch_idxs"),
                    ("candidate", "candidate_mzs_by_cut", "candidate_batch_idxs_by_cut"),
                ):
                    masses = arrays[mz_key][arrays[idx_key] == index]
                    for name, rel, min_mz in (
                        ("flat10", 1e-5, 200.0),
                        ("instrument", tol["rel"], tol["min_mz"]),
                    ):
                        recall, wrecall = _coverage(
                            target_mzs, target_ints, masses, rel, min_mz
                        )
                        row[f"{kind}_{name}_recall"] = recall
                        row[f"{kind}_{name}_wrecall"] = wrecall
                rows.append(row)
            if (batch_number + 1) % 32 == 0:
                print(f"completed {len(rows)} spectra", flush=True)
    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(json.dumps(frame.groupby("inst_type").mean(numeric_only=True).to_dict(), indent=2))


if __name__ == "__main__":
    main()
