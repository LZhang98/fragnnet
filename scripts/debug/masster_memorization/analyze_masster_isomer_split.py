"""Compare checkpoint isomer splits and single-structure cut-class preferences.

Run after ``debug_masster_base_peak_errors.py --all_target_peaks`` on the same
train-molecule sample for each checkpoint.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

try:
    from .debug_masster_base_peak_tables import MoleculeChemistry
except ImportError:
    from debug_masster_base_peak_tables import MoleculeChemistry


def collapse_structures(structures: list[tuple]) -> dict[tuple, float]:
    """Sum duplicate channel probabilities by atom set, H shift, and ion mode."""
    result: dict[tuple, float] = {}
    for atoms, h_shift, mode, probability in structures:
        key = (tuple(atoms), int(h_shift), int(mode))
        result[key] = result.get(key, 0.0) + float(probability)
    return result


def compare_splits(left: pd.DataFrame, right: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Compare common structures in target-matched peaks from two checkpoints."""
    a = left.loc[left.tgt > 0, ["spec_id", "mz", "pred", "tgt", "structures"]].copy()
    b = right.loc[right.tgt > 0, ["spec_id", "mz", "pred", "tgt", "structures"]].copy()
    a["mz_key"] = a.mz.round(6)
    b["mz_key"] = b.mz.round(6)
    matched = a.merge(b, on=["spec_id", "mz_key"], suffixes=("_a", "_b"), validate="one_to_one")
    records = []
    for row in matched.itertuples(index=False):
        pa, pb = collapse_structures(row.structures_a), collapse_structures(row.structures_b)
        common = pa.keys() & pb.keys()
        if len({key[0] for key in common}) < 2 or min(row.pred_a, row.pred_b) < 1e-3:
            continue
        va = np.array([pa[k] for k in sorted(common)])
        vb = np.array([pb[k] for k in sorted(common)])
        cover_a, cover_b = va.sum() / row.pred_a, vb.sum() / row.pred_b
        if min(cover_a, cover_b) < 0.9:
            continue
        va, vb = va / va.sum(), vb / vb.sum()
        midpoint = (va + vb) / 2
        js = float((np.sum(va * np.log2(va / midpoint)) + np.sum(vb * np.log2(vb / midpoint))) / 2)
        records.append({
            "spec_id": row.spec_id, "mz": row.mz_a, "target": row.tgt_a,
            "total_a": row.pred_a, "total_b": row.pred_b,
            "total_rel_diff": abs(row.pred_a - row.pred_b) / max(row.pred_a, row.pred_b),
            "n_common": len(common), "cover_a": cover_a, "cover_b": cover_b,
            "js_bits": js, "top_agrees": int(np.argmax(va) == np.argmax(vb)),
            "l1_half": float(np.abs(va - vb).sum() / 2),
            "top_share_a": float(va.max()), "top_share_b": float(vb.max()),
        })
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError("no shared multi-structure target peaks passed the comparison filters")
    stable = frame.loc[frame.total_rel_diff <= 0.1]
    def summary(part: pd.DataFrame) -> dict:
        if part.empty:
            return {"peaks": 0}
        return {
            "peaks": len(part), "spectra": int(part.spec_id.nunique()),
            "median_total_rel_diff": float(part.total_rel_diff.median()),
            "median_js_bits": float(part.js_bits.median()),
            "p90_js_bits": float(part.js_bits.quantile(0.9)),
            "median_l1_half": float(part.l1_half.median()),
            "top_structure_agreement": float(part.top_agrees.mean()),
            "target_weighted_js_bits": float(np.average(part.js_bits, weights=part.target)),
        }
    confident = stable.loc[(stable.top_share_a >= 0.5) & (stable.top_share_b >= 0.5)]
    return {
        "all": summary(frame), "totals_within_10pct": summary(stable),
        "confident_totals_within_10pct": summary(confident),
    }, frame


def make_features(
    peaks: pd.DataFrame, spectra: pd.DataFrame, mols: pd.Series
) -> tuple[pd.DataFrame, list[str]]:
    """Annotate each structure with cut-class counts and simple geometry controls."""
    mol_id = spectra.set_index("spec_id").mol_id
    rows = []
    cache: dict[int, MoleculeChemistry] = {}
    classes = set()
    for row in peaks.loc[peaks.tgt > 0].itertuples(index=False):
        molecule = int(mol_id[row.spec_id])
        if molecule not in cache:
            cache[molecule] = MoleculeChemistry(mols.loc[molecule])
        structures = collapse_structures(row.structures)
        for (atoms, h_shift, mode), probability in structures.items():
            note = cache[molecule].annotate(atoms, h_shift)
            counts = Counter(note["cut_classes"])
            classes.update(counts)
            rows.append({
                "spec_id": row.spec_id, "mol_id": molecule, "mz": row.mz,
                "target": row.tgt, "prob": probability, "n_structures": len(structures),
                "n_cuts": note["n_cuts"], "frac_atoms": note["frac_atoms"],
                "h_shift_abs": abs(h_shift), "is_precursor": int(note["is_precursor"]),
                "cut_counts": counts,
            })
    table = pd.DataFrame(rows)
    names = sorted(classes)
    for name in names:
        table["cut_" + name] = table.cut_counts.map(lambda c, key=name: c[key]).astype(float)
    return table, ["cut_" + name for name in names]


def cut_preference_read(table: pd.DataFrame, cut_columns: list[str]) -> dict:
    """Cross-fit target-intensity priors on single-structure peaks by molecule."""
    controls = ["n_cuts", "frac_atoms", "h_shift_abs", "is_precursor"]
    singles = table.loc[(table.n_structures == 1) & (table.target >= 1e-3)].copy()
    ambiguous = table.loc[table.n_structures > 1].copy()
    if singles.empty or ambiguous.empty:
        raise ValueError("single-structure or ambiguous target peaks are missing")
    records = []
    single_validation = []
    for fold in (0, 1):
        train = singles.loc[singles.mol_id % 2 != fold].copy()
        test = ambiguous.loc[ambiguous.mol_id % 2 == fold].copy()
        test_single = singles.loc[singles.mol_id % 2 == fold].copy()
        if train.empty or test.empty:
            continue
        y = np.log(train.target.to_numpy())
        y -= train.groupby("spec_id").target.transform(lambda v: np.log(v).mean()).to_numpy()
        fit = {}
        validation = {"fold": fold, "single_peaks": len(test_single)}
        y_single = np.log(test_single.target.to_numpy())
        y_single -= test_single.groupby("spec_id").target.transform(
            lambda v: np.log(v).mean()
        ).to_numpy()
        for label, columns in (("control", controls), ("cut", controls + cut_columns)):
            x = train[columns]
            x = x - x.groupby(train.spec_id).transform("mean")
            model = Ridge(alpha=10.0, fit_intercept=False).fit(x.to_numpy(), y)
            fit[label] = model.predict(test[columns].to_numpy())
            x_single = test_single[columns]
            x_single = x_single - x_single.groupby(test_single.spec_id).transform("mean")
            residual = y_single - model.predict(x_single.to_numpy())
            validation[label + "_r2"] = float(1 - residual @ residual / (y_single @ y_single))
        single_validation.append(validation)
        test["control_score"] = fit["control"]
        test["cut_score"] = fit["cut"]
        for (_spec_id, _mz), group in test.groupby(["spec_id", "mz"], sort=False):
            if len(group) < 2 or group.prob.sum() < 1e-3:
                continue
            model_top = int(np.argmax(group.prob.to_numpy()))
            records.append({
                "fold": fold, "mol_id": int(group.mol_id.iloc[0]),
                "n": len(group), "target": float(group.target.iloc[0]),
                "control_hit": int(np.argmax(group.control_score.to_numpy()) == model_top),
                "cut_hit": int(np.argmax(group.cut_score.to_numpy()) == model_top),
                "random_hit": 1 / len(group),
            })
    result = pd.DataFrame(records)
    if result.empty:
        raise ValueError("no ambiguous peaks passed cut-preference filters")
    per_molecule = result.groupby("mol_id").agg(
        effect_sum=("cut_hit", "sum"), baseline_sum=("control_hit", "sum"),
        count=("cut_hit", "size"),
    )
    rng = np.random.default_rng(0)
    drawn = rng.integers(len(per_molecule), size=(1000, len(per_molecule)))
    numerators = (per_molecule.effect_sum - per_molecule.baseline_sum).to_numpy()
    denominators = per_molecule["count"].to_numpy()
    boot = numerators[drawn].sum(axis=1) / denominators[drawn].sum(axis=1)
    return {
        "single_structure_training_peaks": len(singles),
        "ambiguous_test_peaks": len(result),
        "molecules": int(ambiguous.mol_id.nunique()),
        "top_hit_random": float(result.random_hit.mean()),
        "top_hit_controls": float(result.control_hit.mean()),
        "top_hit_with_cut_classes": float(result.cut_hit.mean()),
        "cut_increment": float((result.cut_hit - result.control_hit).mean()),
        "molecule_bootstrap_cut_increment_ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
        "target_weighted_cut_increment": float(np.average(
            result.cut_hit - result.control_hit, weights=result.target
        )),
        "by_fold": result.groupby("fold")[["control_hit", "cut_hit"]].mean().to_dict("index"),
        "single_structure_validation": single_validation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i1", required=True)
    parser.add_argument("--other", required=True)
    parser.add_argument("--mol_fp", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    i1, other = Path(args.i1), Path(args.other)
    spectra = pd.read_pickle(i1)
    peaks_i1 = pd.read_pickle(i1.with_suffix(".peaks.pkl"))
    peaks_other = pd.read_pickle(other.with_suffix(".peaks.pkl"))
    comparison, rows = compare_splits(peaks_i1, peaks_other)
    mols = pd.read_pickle(args.mol_fp).set_index("mol_id").mol
    # Three evenly spaced spectra per molecule keep the chemical read CPU-bounded.
    sampled = spectra.groupby("mol_id", sort=False).spec_id.apply(
        lambda ids: ids.iloc[np.unique(np.linspace(0, len(ids) - 1, 3, dtype=int))].tolist()
    )
    keep_ids = {spec_id for group in sampled for spec_id in group}
    chem_peaks = peaks_i1.loc[peaks_i1.spec_id.isin(keep_ids)]
    features, cut_columns = make_features(chem_peaks, spectra, mols)
    result = {
        "inputs": {"i1": str(i1), "other": str(other), "mol_fp": args.mol_fp},
        "split_comparison": comparison,
        "chemical_read_spectra": len(keep_ids),
        "cut_preference": cut_preference_read(features, cut_columns),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    rows.to_pickle(output.with_suffix(".matched.pkl"))


if __name__ == "__main__":
    main()
