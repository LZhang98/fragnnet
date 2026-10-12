"""Does our DAG explain NIST-annotated peaks with NIST's formula, or with a different one?

Peak coverage is normally measured by m/z matching, which cannot tell a correct assignment from
a coincidental one. NIST ships an expert formula per peak for 68% of our spectra, so the two can
be separated:

  uncovered        - no DAG formula lands within tolerance of the peak at all
  identity match   - a DAG formula matches AND NIST's annotated formula is among the matches
  identity MISMATCH- a DAG formula matches but NIST's annotated formula is not in the DAG's
                     support, i.e. the peak scores as explained while the explanation is wrong

The mismatch fraction is the part of our coverage that is credit for the wrong molecule.

The mismatch bucket is split further. A NIST formula whose heavy-atom SKELETON the DAG can
build, differing only in hydrogen count, is an h-window or ion-convention difference rather than
missing chemistry; a skeleton the DAG cannot build at all is the real residual.

NIST product formulas are ion formulas as written: their neutral-atom mass matches the annotated
exact m/z to within one electron mass on 100% of a 121,703-peak sample, which is what licenses
the `neutral = ion - H` conversion below. They are assignments, not ground truth -- 0.72% carry a
chemically implausible RDBE -- so treat the mismatch as a disagreement to investigate rather than
as a verdict.

Results and their interpretation:
``docs/findings/unused_information_inventory_2026_09_22.md``.

Usage:
    python scripts/debug/dag/debug_annotation_vs_dag_identity.py
"""

import numpy as np
import pandas as pd

from fragnnet.models.fragnnet.frag_utils import load_frag_d
from fragnnet.utils.formula_utils import parse_formula

FRAG_DP = "data/frag/nist20_ma_mi_d3_h4_isoFalse_smpFalse/dags"
PROTON = 1.007276
MASSES = {"C": 12.0, "H": 1.0078250319, "N": 14.0030740052, "O": 15.9949146221,
          "S": 31.97207069, "P": 30.97376151, "F": 18.99840320, "Cl": 34.96885271,
          "Br": 78.9183376, "I": 126.904473, "Si": 27.9769265327, "Se": 79.9165196,
          "B": 11.0093055, "As": 74.9215942, "Sn": 119.9021947, "Pb": 207.9766521,
          "Hg": 201.970617}
ELEMENTS = tuple(MASSES)


def key_of(counts):
    return tuple(counts.get(e, 0) for e in ELEMENTS)


def mass_of(counts):
    return sum(MASSES[e] * n for e, n in counts.items() if e in MASSES)


def main(num_molecules=400, ppm=10.0):
    ann = pd.read_pickle("data/proc/nist20/ann_df.pkl")
    ann = ann[ann["prec_type"] == "[M+H]+"]
    spec = pd.read_pickle("data/proc/nist20/spec_df.pkl")
    peaks_by_id = dict(zip(spec["spec_id"].astype(str), spec["peaks"]))
    del spec

    rng = np.random.default_rng(0)
    mol_ids = ann["mol_id"].unique()
    mol_ids = rng.choice(mol_ids, size=min(num_molecules, len(mol_ids)), replace=False)
    by_mol = dict(ann[ann["mol_id"].isin(mol_ids)].groupby("mol_id").__iter__())

    cache: dict = {}
    stats = np.zeros(4)          # uncovered, match, H-only mismatch, skeleton mismatch
    weighted = np.zeros(4)       # same, weighted by peak intensity
    missing_examples: dict = {}
    molecules_done = 0
    for mol_id in mol_ids:
        try:
            frag = load_frag_d(int(mol_id), FRAG_DP, is_compressed=True)
        except Exception:
            continue
        if frag is None:
            continue
        # 1. The DAG's formula support, as protonated ion m/z.
        support_keys, support_mz = [], []
        for formula in frag["idx_to_formula"].values():
            if not formula:
                continue
            if formula not in cache:
                try:
                    cache[formula] = parse_formula(formula)
                except Exception:
                    cache[formula] = None
            counts = cache[formula]
            if counts is None:
                continue
            support_keys.append(key_of(counts))
            support_mz.append(mass_of(counts) + PROTON)
        if not support_mz:
            continue
        support_mz = np.asarray(support_mz)
        order = np.argsort(support_mz)
        support_mz = support_mz[order]
        support_keys = [support_keys[i] for i in order]
        support_set = set(support_keys)
        heavy_index = [i for i, e in enumerate(ELEMENTS) if e != "H"]
        heavy_set = {tuple(k[i] for i in heavy_index) for k in support_keys}
        molecules_done += 1

        # 2. Score every annotated peak of every spectrum of this molecule.
        group = by_mol[mol_id]
        for spec_id, products, mzs, isotopes in zip(
            group["spec_id"].astype(str), group["ann_products"],
            group["ann_peak_mzs"], group["ann_isotopes"],
        ):
            peaks = peaks_by_id.get(spec_id)
            if not peaks:
                continue
            peak_arr = np.asarray(peaks, float)
            total = peak_arr[:, 1].sum()
            if total <= 0:
                continue
            intensity_by_mz = {round(mz, 2): i for mz, i in peaks}
            for product, mz, isotope in zip(products, mzs, isotopes):
                if isotope != 0:
                    continue
                mz = float(mz)
                if product not in cache:
                    try:
                        cache[product] = parse_formula(product)
                    except Exception:
                        cache[product] = None
                counts = cache[product]
                if counts is None:
                    continue
                # NIST products are ION formulas; the DAG stores neutral fragments.
                neutral = dict(counts)
                neutral["H"] = neutral.get("H", 0) - 1
                if neutral["H"] < 0:
                    continue
                intensity = intensity_by_mz.get(round(mz, 2), 0.0) / total
                tolerance = mz * ppm * 1e-6
                lo = np.searchsorted(support_mz, mz - tolerance)
                hi = np.searchsorted(support_mz, mz + tolerance)
                if hi <= lo:
                    bucket = 0
                elif key_of(neutral) in support_set:
                    bucket = 1
                else:
                    full = key_of(neutral)
                    skeleton = tuple(full[i] for i in heavy_index)
                    # A skeleton the DAG can build, differing only in hydrogen count, is an
                    # h-window or ion-convention difference, not missing chemistry.
                    bucket = 2 if skeleton in heavy_set else 3
                    if bucket == 3:
                        if len(missing_examples) < 12:
                            missing_examples.setdefault(product, 0)
                        if product in missing_examples:
                            missing_examples[product] += 1
                stats[bucket] += 1
                weighted[bucket] += intensity

    labels = [
        "no DAG formula within tolerance",
        "DAG matches, NIST formula IS in support",
        "mismatch, but the heavy skeleton IS reachable (H only)",
        "mismatch, heavy skeleton NOT reachable (real)",
    ]
    print(f"molecules={molecules_done}  annotated monoisotopic peaks={int(stats.sum()):,}\n")
    print(f"{'outcome':46s} {'peaks':>10s} {'intensity':>11s}")
    for label, count, weight in zip(labels, stats, weighted):
        print(f"{label:46s} {count / stats.sum():10.3f} {weight / weighted.sum():11.3f}")
    print("\nexample NIST formulas whose skeleton the DAG cannot build:",
          sorted(missing_examples.items(), key=lambda kv: -kv[1])[:8])


if __name__ == "__main__":
    main()
