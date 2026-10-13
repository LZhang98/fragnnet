"""Per-spectrum share of Spectraverse signal that the labelled structure can explain.

A peak is explained when it lies within the instrument tolerance (``TOL`` in
``clean_train_split.py``) of a precursor-bounded subformula ion of the labelled molecule and
adduct (``_subformula_masses``, charge-migration ions included). Per spectrum:

- ``frac_intensity``: summed intensity of explained peaks / summed intensity of all peaks
  (stored, base-peak-normalized intensities, linear).
- ``frac_peaks``: explained peaks / all peaks.
- ``ceiling``: sqrt of the squared-intensity share, as in ``subformula_ceiling.csv``.

    python preproc_scripts/spectraverse/subformula_explained.py
"""

from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from clean_train_split import subformula_hits  # noqa: E402


def _explained(args) -> tuple[float, float, float]:
    hits = subformula_hits(*args)
    if hits is None:
        return np.nan, np.nan, np.nan
    ints, hit = hits
    if not (ints > 0).any():
        return 0.0, 0.0, 0.0
    return (
        float(ints[hit].sum() / ints.sum()),
        float(hit.mean()),
        float(np.sqrt((ints[hit] ** 2).sum() / (ints**2).sum())),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec-fp", default="data/proc/spectraverse/spec_df.pkl")
    parser.add_argument("--mol-fp", default="data/proc/spectraverse/mol_df.pkl")
    parser.add_argument("--out-fp", default="data/proc/spectraverse/subformula_explained.csv")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    spec = pd.read_pickle(args.spec_fp)
    formula = pd.read_pickle(args.mol_fp).set_index("mol_id")["formula"]
    spec = spec.sort_values(["mol_id", "prec_type"])  # keeps _subformula_masses cache hits
    jobs = zip(spec.peaks, formula.loc[spec.mol_id].values, spec.prec_type, spec.inst_type)
    with Pool(args.workers) as pool:
        values = pool.map(_explained, jobs, chunksize=512)
    out = pd.DataFrame(values, columns=["frac_intensity", "frac_peaks", "ceiling"])
    out.insert(0, "spec_id", spec["spec_id"].astype(np.int64).to_numpy())
    out.to_csv(args.out_fp, index=False)
    print(out.describe().round(3).to_string())


if __name__ == "__main__":
    main()
