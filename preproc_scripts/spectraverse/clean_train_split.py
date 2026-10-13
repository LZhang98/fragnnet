"""Build Spectraverse fold splits whose TRAIN part drops spectra no subformula explains.

Per spectrum, the subformula ceiling is sqrt(sum of squared intensity of peaks within the
instrument tolerance of a precursor-bounded subformula ion / sum of all squared intensity),
the peak-level form of the ``sub_inst`` oracle in
``scripts/debug/spectraverse/spectraverse_oracle_ceiling.py`` (that one bins at 0.01 Da after model
preprocessing). Train spectra with a ceiling at or below ``--min-ceiling`` are removed;
val and test ids are copied unchanged, so held-out scores stay comparable with the source
split. See docs/findings/masster/spectraverse_oracle_ceiling_2026_10_03.md.

    python preproc_scripts/spectraverse/clean_train_split.py --folds 0 1
"""

from __future__ import annotations

import argparse
import json
import shutil
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from fragnnet.utils.clean_spectra_utils import _subformula_masses

# inst_type_loss_tol of config/masster/spectraverse/cv*.yml
TOL = {
    "ORB": (1e-5, 200.0),
    "QTOF": (2e-5, 750.0),
    "Other": (2e-5, 500.0),
    "NA": (2e-5, 500.0),
}


def subformula_hits(peaks, formula, prec_type, inst):
    """Intensities and a per-peak mask: within ``TOL[inst]`` of a subformula ion.

    Returns ``None`` when the formula is unsupported, and an all-False mask when there are no
    candidate masses.
    """
    mzs = np.array([p[0] for p in peaks], dtype=np.float64)
    ints = np.array([p[1] for p in peaks], dtype=np.float64)
    try:
        masses = _subformula_masses(formula, prec_type)
    except ValueError:
        return None
    if not len(masses):
        return ints, np.zeros(len(mzs), dtype=bool)
    rel, min_mz = TOL[inst]
    right = np.searchsorted(masses, mzs)
    left = np.maximum(right - 1, 0)
    right = np.minimum(right, len(masses) - 1)
    dist = np.minimum(np.abs(masses[left] - mzs), np.abs(masses[right] - mzs))
    return ints, dist <= rel * np.maximum(mzs, min_mz)


def _ceiling(args):
    hits = subformula_hits(*args)
    if hits is None:
        return np.nan
    ints, hit = hits
    if not (ints > 0).any():
        return 0.0
    return float(np.sqrt((ints[hit] ** 2).sum() / (ints**2).sum()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec-fp", default="data/proc/spectraverse/spec_df.pkl")
    parser.add_argument("--mol-fp", default="data/proc/spectraverse/mol_df.pkl")
    parser.add_argument("--src-split-dp", default="data/split/spectraverse_benchmark")
    parser.add_argument("--dst-split-dp", default="data/split/spectraverse_benchmark_subceil05")
    parser.add_argument("--ceiling-fp", default="data/proc/spectraverse/subformula_ceiling.csv")
    parser.add_argument("--min-ceiling", type=float, default=0.5)
    parser.add_argument("--folds", type=int, nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    ceiling_fp = Path(args.ceiling_fp)
    if ceiling_fp.exists():
        ceil = pd.read_csv(ceiling_fp)
    else:
        spec = pd.read_pickle(args.spec_fp)
        formula = pd.read_pickle(args.mol_fp).set_index("mol_id")["formula"]
        spec = spec.sort_values(["mol_id", "prec_type"])  # keeps _subformula_masses cache hits
        jobs = zip(spec.peaks, formula.loc[spec.mol_id].values, spec.prec_type, spec.inst_type)
        with Pool(args.workers) as pool:
            values = pool.map(_ceiling, jobs, chunksize=512)
        ceil = pd.DataFrame(
            {"spec_id": spec.spec_id.values, "prec_type": spec.prec_type.values,
             "inst_type": spec.inst_type.values, "subformula_ceiling": values}
        )
        ceil.to_csv(ceiling_fp, index=False)
    drop = set(ceil.spec_id[ceil.subformula_ceiling <= args.min_ceiling])
    stats = {"min_ceiling": args.min_ceiling, "n_spectra": len(ceil),
             "n_unsupported": int(ceil.subformula_ceiling.isna().sum()),
             "n_below": len(drop), "folds": {}}
    for fold in args.folds:
        src = Path(args.src_split_dp) / f"cv{fold}"
        dst = Path(args.dst_split_dp) / f"cv{fold}"
        dst.mkdir(parents=True, exist_ok=True)
        train = pd.read_csv(src / "train_ids.csv")
        kept = train[~train.spec_id.isin(drop)]
        kept.to_csv(dst / "train_ids.csv", index=False)
        for name in ("val_ids.csv", "test_ids.csv"):
            shutil.copyfile(src / name, dst / name)
        stats["folds"][f"cv{fold}"] = {
            "train_before": len(train), "train_after": len(kept),
            "train_dropped": len(train) - len(kept),
            "train_mols_before": int(train.mol_id.nunique()),
            "train_mols_after": int(kept.mol_id.nunique()),
        }
    (Path(args.dst_split_dp) / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
