"""Oracle cos_sim_0.01 ceilings for Spectraverse MASSter on sampled validation spectra.

Each oracle keeps the true (m/z, intensity) of the target peaks a rule calls reachable and
zeroes the rest, then scores that against the full target with the training metric's binning
(``batched_bin_func`` + ``cos_sim_helper``). The ceilings are:

- ``opt_emit``: the repo's ``opt_cos_sim`` over the model's emitted 0.01 Da bins.
- ``cand_inst`` / ``cand_005``: a candidate mass within instrument tolerance / within 0.05 Da.
- ``cand_inst_iso``: ``cand_inst`` plus peaks 1 or 2 x 1.003355 above a candidate.
- ``sub_inst`` / ``sub_005``: any precursor-bounded subformula ion (``_subformula_masses``,
  model independent) within instrument tolerance / 0.05 Da.

``model`` is the model's own ``cos_sim_0.01``, recomputed here and checked against
``inference_step``'s value.
"""

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
from fragnnet.utils.clean_spectra_utils import _subformula_masses
from fragnnet.utils.setup_utils import get_dataset_cls
from fragnnet.utils.spec_similarity import cos_sim_helper, opt_cos_sim_helper
from fragnnet.utils.spec_utils import batched_bin_func

C13 = 1.003355
BATCH = 4


def _near(target_mzs, masses, tol):
    """Boolean mask of target peaks within ``tol`` (array, Da) of any mass."""
    masses = np.sort(np.asarray(masses, dtype=np.float64))
    if not len(masses):
        return np.zeros(len(target_mzs), dtype=bool)
    right = np.searchsorted(masses, target_mzs)
    left = np.maximum(right - 1, 0)
    right = np.minimum(right, len(masses) - 1)
    dist = np.minimum(np.abs(masses[left] - target_mzs), np.abs(masses[right] - target_mzs))
    return dist <= tol


def _binned_cos(true_mzs, true_ints, keep, mz_max, agg):
    """cos_sim_0.01 between a target and its own peaks restricted to ``keep``."""
    if not keep.any():
        return 0.0
    t = [th.as_tensor(x) for x in (true_mzs, true_ints)]
    zeros = th.zeros(len(true_mzs), dtype=th.long)
    tb = batched_bin_func(t[0], t[1], zeros, mz_max=mz_max, mz_bin_res=0.01, agg=agg, sparse=True)
    k = th.as_tensor(keep)
    pb = batched_bin_func(
        t[0][k], t[1][k], zeros[k], mz_max=mz_max, mz_bin_res=0.01, agg=agg, sparse=True
    )
    return float(cos_sim_helper(*tb, *pb)[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=1024)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--config", help="custom config (default: the SPV fold config)")
    args = parser.parse_args()
    custom = args.config or f"config/spectraverse/masster/cv{args.fold}.yml"
    config = load_config("config/template.yml", custom, [])
    config["num_workers"] = 0
    config["masster_candidate_backend"] = "cpu"
    device = th.device("cpu")
    source = pd.read_pickle(config["spec_fp"])
    ids = pd.read_csv(Path(config["split_dp"]) / f"{args.split}_ids.csv")["spec_id"]
    selected = source[source.spec_id.isin(ids)].sample(args.sample_size, random_state=20261001)
    selected = selected.sort_values("spec_id")
    molecules = pd.read_pickle(config["mol_fp"])
    subset = args.output.parent / f"subset_{args.split}_cv{args.fold}_{args.sample_size}"
    subset.mkdir(parents=True, exist_ok=True)
    selected.to_pickle(subset / "spec_df.pkl")
    molecules[molecules.mol_id.isin(selected.mol_id)].to_pickle(subset / "mol_df.pkl")
    config["spec_fp"] = str(subset / "spec_df.pkl")
    config["mol_fp"] = str(subset / "mol_df.pkl")
    formula = molecules.set_index("mol_id")["formula"]
    model = init_model_from_ckpt(str(args.checkpoint), config, device)
    mz_max = float(model.hparams.mz_max)
    agg = "sum" if model.hparams.sum_ints else "amax"
    dataset = get_dataset_cls(config["model_type"])(split=args.split, **config)
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=False, collate_fn=dataset.get_collate_fn())
    rows = []
    with th.inference_mode():
        for batch_number, batch in enumerate(loader):
            batch = model.transfer_batch_to_device(batch, device, dataloader_idx=0)
            out = model.inference_step(batch, metrics="val")
            tm, ti, tb = out["true_mzs"], th.exp(out["true_logprobs"]), out["true_batch_idxs"]
            pm, pi, pbi = out["pred_mzs"], out["pred_ints"], out["pred_batch_idxs"]
            keep = pm < mz_max
            true_bins = batched_bin_func(tm, ti, tb, mz_max=mz_max, mz_bin_res=0.01, agg=agg, sparse=True)
            pred_bins = batched_bin_func(
                pm[keep], pi[keep], pbi[keep], mz_max=mz_max, mz_bin_res=0.01, agg=agg, sparse=True
            )
            model_cos = cos_sim_helper(*true_bins, *pred_bins)
            opt_emit = opt_cos_sim_helper(*true_bins, *pred_bins)
            cand = out["candidate_mzs_by_cut"].numpy()
            cand_b = out["candidate_batch_idxs_by_cut"].numpy()
            for i in range(int(batch["batch_size"])):
                spec = dataset.spec_df.iloc[batch_number * BATCH + i]
                tol = config["inst_type_loss_tol"][spec.inst_type]
                mask = (tb == i).numpy()
                mzs, ints = tm.numpy()[mask].astype(np.float64), ti.numpy()[mask].astype(np.float64)
                tol_inst = tol["rel"] * np.maximum(mzs, tol["min_mz"])
                tol_005 = np.full_like(mzs, 0.05)
                masses = cand[cand_b == i]
                try:
                    sub = _subformula_masses(formula[spec.mol_id], spec.prec_type)
                except ValueError:
                    sub = np.array([])
                reach = {
                    "cand_inst": _near(mzs, masses, tol_inst),
                    "cand_005": _near(mzs, masses, tol_005),
                    "sub_inst": _near(mzs, sub, tol_inst),
                    "sub_005": _near(mzs, sub, tol_005),
                }
                reach["cand_inst_iso"] = (
                    reach["cand_inst"]
                    | _near(mzs, masses + C13, tol_inst)
                    | _near(mzs, masses + 2 * C13, tol_inst)
                )
                row = {
                    "spec_id": int(spec.spec_id),
                    "inst_type": spec.inst_type,
                    "prec_type": spec.prec_type,
                    "has_ce": bool(np.any(pd.notna(spec.ace)) or np.any(pd.notna(spec.nce))),
                    "n_peaks": len(mzs),
                    "model": float(model_cos[i]),
                    "model_ref": float(out["cos_sim_0.01"][i]),
                    "opt_emit": float(opt_emit[i]),
                }
                for name, r in reach.items():
                    row[name] = _binned_cos(mzs, ints, r, mz_max, agg)
                    row[f"{name}_wrecall"] = float(ints[r].sum() / ints.sum())
                rows.append(row)
            if (batch_number + 1) % 32 == 0:
                print(f"completed {len(rows)} spectra", flush=True)
    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print("max |model - model_ref|:", float((frame.model - frame.model_ref).abs().max()))
    cols = ["model", "opt_emit", "cand_inst", "cand_inst_iso", "cand_005", "sub_inst", "sub_005"]
    print(json.dumps(frame[cols].mean().round(4).to_dict(), indent=2))


if __name__ == "__main__":
    main()
