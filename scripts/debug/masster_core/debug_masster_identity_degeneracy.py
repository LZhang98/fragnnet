"""What can MASSter's losses NOT see? Substructure identity.

MASSter is supervised entirely through m/z. ``_proposal_coverage_loss`` rewards a proposed
fragment whose m/z falls within tolerance of an observed peak and never checks which fragment it
is; the intensity loss is a spectral objective at instrument tolerance. So wherever several
candidates share an m/z, the objective is exactly indifferent between them and the model's choice
is settled by inductive bias rather than by evidence.

Two degeneracies are measured, and they are very different in size:

  composition   two DIFFERENT formulas inside one tolerance window. NIST's expert annotation
                would resolve these, so this bounds what formula-level supervision could buy.
  substructure  the SAME formula reachable as several different connected fragments. Identical
                m/z by construction, so no formula-level label resolves it -- only a structural
                teacher does.

Both are computed over the depth-3 DAG's support, which is a SUBSET of what MASSter's
unrestricted multi-cut masks can build. Composition ambiguity is therefore a lower bound for
MASSter, and so is substructure degeneracy.

Results and their interpretation:
``docs/findings/masster/masster_identity_supervision_2026_09_22.md``.

Usage:
    python scripts/debug/masster_core/debug_masster_identity_degeneracy.py
"""

import numpy as np
import pandas as pd

from fragnnet.utils.dag_format_utils import load_frag_d
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


def main(num_molecules=300, ppm=10.0):
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
    ratios = []
    ambiguous = matched = 0.0
    ambiguous_int = matched_int = 0.0
    for mol_id in mol_ids:
        try:
            frag = load_frag_d(int(mol_id), FRAG_DP, is_compressed=True)
        except Exception:
            continue
        if frag is None:
            continue

        # 1. Substructure degeneracy: DAG nodes are distinct connected fragments; the h-delta-0
        #    bucket names their formulas, so the ratio counts substructures per formula.
        h_deltas = frag["idx_by_h_delta"]
        base_formulas = {frag["idx_to_formula"][i] for i in h_deltas[len(h_deltas) // 2]}
        if base_formulas:
            ratios.append(frag["dag_num_nodes"] / len(base_formulas))

        # 2. Composition ambiguity: distinct formulas inside one tolerance window.
        keys, support = [], []
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
            keys.append(key_of(counts))
            support.append(sum(MASSES[e] * n for e, n in counts.items() if e in MASSES) + PROTON)
        if not support:
            continue
        support = np.asarray(support)
        order = np.argsort(support)
        support = support[order]
        keys = [keys[i] for i in order]

        for spec_id, mzs, isotopes in zip(
            by_mol[mol_id]["spec_id"].astype(str), by_mol[mol_id]["ann_peak_mzs"],
            by_mol[mol_id]["ann_isotopes"],
        ):
            peaks = peaks_by_id.get(spec_id)
            if not peaks:
                continue
            arr = np.asarray(peaks, float)
            total = arr[:, 1].sum()
            if total <= 0:
                continue
            intensity_by_mz = {round(mz, 2): i for mz, i in peaks}
            for mz, isotope in zip(mzs, isotopes):
                if isotope != 0:
                    continue
                mz = float(mz)
                tolerance = mz * ppm * 1e-6
                lo = np.searchsorted(support, mz - tolerance)
                hi = np.searchsorted(support, mz + tolerance)
                if hi <= lo:
                    continue
                intensity = intensity_by_mz.get(round(mz, 2), 0.0) / total
                matched += 1
                matched_int += intensity
                if len({keys[i] for i in range(lo, hi)}) > 1:
                    ambiguous += 1
                    ambiguous_int += intensity

    ratios = np.asarray(ratios)
    print(f"molecules={len(ratios)}   annotated peaks with a reachable formula={int(matched):,}\n")
    print("COMPOSITION ambiguity (a formula label would resolve it):")
    print(f"  peaks with >=2 distinct formulas in tolerance: {ambiguous / matched:.3f}")
    print(f"  share of matched intensity:                    {ambiguous_int / matched_int:.3f}\n")
    print("SUBSTRUCTURE degeneracy (no formula label resolves it):")
    print(f"  distinct connected fragments per formula: mean {ratios.mean():.2f}"
          f"  p50 {np.median(ratios):.2f}  p90 {np.percentile(ratios, 90):.2f}"
          f"  max {ratios.max():.2f}")
    print(f"  molecules averaging >=2 fragments per formula: {100 * (ratios >= 2).mean():.1f}%")


if __name__ == "__main__":
    main()
