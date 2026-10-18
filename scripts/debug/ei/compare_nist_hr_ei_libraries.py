"""Compare measured HR-EI reference spectra with NIST23 nominal EI spectra of the same molecules.

For every HR-EI record whose 2D connectivity has NIST23 EI spectra, compute nominal linear cosines:
HR vs NIST23 (library agreement), model vs NIST23 and model vs HR, plus NIST23 replicate agreement.
HR spectra and exact-mass predictions are rounded to integer m/z and summed. ``window`` cuts every
spectrum to the source acquisition range; ``full`` keeps all peaks.

    /home/feiw/miniconda3/envs/FRAGNNET-GPU/bin/python -m scripts.debug.ei.compare_nist_hr_ei_libraries \
        --eval_dp output/hr_ei_e5_eugy8s9n_eval_2026_10_18/natural_exact --output output/hr_ei_vs_nist23_2026_10_18
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.debug.ei.evaluate_hr_ei_checkpoint import aggregate, score

# Lower bound: declared (HREI-MSDB) or the lowest peak in any record of the source (RECETOX).
WINDOWS = {"hrei_msdb": (40, 500), "recetox_exposome": (67, np.inf), "recetox_metabolome": (70, np.inf)}


def nominal(mz: np.ndarray, intensity: np.ndarray, window: tuple[float, float] | None) -> tuple:
    """Integer-mass spectrum, optionally restricted to ``window``."""
    mz, intensity = aggregate(np.rint(mz), intensity, decimals=0)
    mz = np.rint(mz)  # weighted aggregation can leave 195.99999999
    if window is not None:
        keep = (mz >= window[0]) & (mz <= window[1])
        mz, intensity = mz[keep], intensity[keep]
    return mz, intensity


def cosine(a: tuple, b: tuple) -> float:
    return score(a[0], a[1], b[0], b[1], 0, 0)["cosine"] if len(a[0]) and len(b[0]) else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval_dp", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    sel = pd.read_csv(args.eval_dp / "selected_records.csv").set_index("spec_id")
    hr_spec = pd.read_pickle(args.eval_dp / "spec_df.pkl").set_index("spec_id")
    pred = np.load(args.eval_dp / "cache/exact_predictions.npz")
    mol = pd.read_pickle("data/proc/nist23_ei/mol_df.pkl")[["mol_id", "inchikey_s"]]
    nist = pd.read_pickle("data/proc/nist23_ei/spec_df.pkl")[["spec_id", "mol_id", "peaks"]].merge(mol, on="mol_id")
    nist["ck"] = nist["inchikey_s"].str[:14]
    nist = nist[nist["ck"].isin(set(sel["connectivity_key"]))]
    by_key = {k: [np.asarray(p, dtype=float) for p in g["peaks"]] for k, g in nist.groupby("ck")}

    rows = []
    for sid, rec in sel.iterrows():
        refs = by_key.get(rec["connectivity_key"])
        if not refs:
            continue
        hr = np.asarray(hr_spec.loc[sid, "peaks"], dtype=float)
        row = {"spec_id": sid, "source": rec["source"], "name": rec["name"], "n_nist": len(refs),
               "nominal_train_overlap": rec["nominal_train_overlap"]}
        for label, window in (("window", WINDOWS[rec["source"]]), ("full", None)):
            h = nominal(hr[:, 0], hr[:, 1], window)
            p = nominal(pred[f"{sid}_mz"], pred[f"{sid}_int"], window)
            ns = [nominal(r[:, 0], r[:, 1], window) for r in refs]
            row[f"hr_vs_nist_{label}"] = np.mean([cosine(h, n) for n in ns])
            row[f"hr_vs_nist_best_{label}"] = max(cosine(h, n) for n in ns)
            row[f"model_vs_nist_{label}"] = np.mean([cosine(p, n) for n in ns])
            row[f"model_vs_hr_{label}"] = cosine(p, h)
            pairs = [cosine(a, b) for a, b in combinations(ns, 2)]
            row[f"nist_replicates_{label}"] = np.mean(pairs) if pairs else np.nan
        rows.append(row)

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "pairs.csv", index=False)
    cols = [c for c in frame.columns if c.endswith(("_window", "_full"))]
    summary = frame.groupby("source")[cols].mean().T.round(4)
    summary.loc["records"] = frame.groupby("source").size()
    summary.loc["records_with_nist_replicates"] = frame.groupby("source")["nist_replicates_window"].count()
    summary.loc["records_total"] = sel.groupby("source").size()
    summary.to_csv(args.output / "summary.csv")
    print(summary.to_string())


if __name__ == "__main__":
    main()
