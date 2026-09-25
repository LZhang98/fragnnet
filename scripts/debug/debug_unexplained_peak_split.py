"""Split the unexplained intensity of a merged target into "not this molecule" and "coverage gap".

MASSter reaches 89.1% of validation intensity with its candidate set (`val_wrecall`), and
the 2026-09-22 overfit read-out established that the remaining ~11% is the only true coverage
ceiling. That residual has two populations needing opposite fixes:

- A peak whose m/z admits **no** precursor-bounded, nonnegative-RDBE subformula cannot come
  from this molecule at all. It is co-isolation, background or noise, and the model is being
  graded against it. That is what an escape channel is for.
- A peak that **does** admit a feasible subformula is a real analyte fragment the candidate
  enumeration misses. An escape would only excuse it; the fix is enumeration or rearrangement
  chemistry.

This probe measures that split, model-free and without a checkpoint, on exactly the target a
run trains against: the same `spec_params` (so the same cleanup and `tolerance_consensus`
merge) and the same split. Feasibility reuses production `_subformula_masses`, the same
enumeration `filter_subformula_peaks` applies, at per-instrument tolerance.

Because every enumerated candidate is by construction a precursor-bounded subformula, the
feasible fraction is an upper bound on what any candidate set could ever reach. The gap
between it and a run's measured `val_wrecall` is the coverage population; one minus it is the
escape population.

It also reports how concentrated the infeasible mass is across spectra, which decides the
shape of the escape: mass clustered in a few spectra is addressable with a per-spectrum gate,
mass spread evenly over many spectra needs a per-peak responsibility.

The probe also classifies every peak against the precursor m/z. Nothing in the data path
removes peaks above the precursor, so they are all in the training target; but most of that
intensity is expected to be the precursor's own isotopologues, which are real analyte signal a
monoisotopic candidate set can never match. Separating the isotope ladder from the rest says
whether an above-precursor cut would delete noise or delete signal.

The same logic applies below the precursor: a fragment's own M+1/M+2 isotopologue is real
analyte signal that a monoisotopic candidate set can never match, and it is indistinguishable
from a contaminant to any test that only asks "does this m/z have a subformula". The probe
therefore also asks, for every infeasible peak, whether a *feasible* peak sits one or two
neutron masses below it -- the signature of an isotopologue of an explainable fragment.

``--null_control`` measures how much of the verdict is chemistry rather than arithmetic. A
large precursor admits so many subformulas that almost any m/z lands within tolerance of one,
so the infeasible fraction under the *correct* formula is only meaningful against the fraction
under a *wrong* formula of similar precursor mass. The control re-scores every spectrum against
a donor molecule drawn from its own precursor-mass band. A donor rate close to the true rate
means the test is nearly vacuous and the split below cannot be trusted; a much higher donor
rate means the test discriminates and the infeasible population is real.

Usage:
    python scripts/debug/debug_unexplained_peak_split.py \
        --custom_fp config/masster/ma_mi_merged/nist20_ma_mi_merged_fast_batch512_cuda_lb64_c4a_base_s1.yml \
        --split val \
        --measured_wrecall 0.8911
"""

import argparse
import functools
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from fragnnet.dataset.base_dataset import BaseDataset  # noqa: E402
from fragnnet.runner import load_config  # noqa: E402
from fragnnet.utils.clean_spectra_utils import (  # noqa: E402
    INSTRUMENT_TOLERANCES,
    _subformula_masses,
)


@functools.cache
def feasible_masses(formula: str, prec_type: str) -> np.ndarray:
    """Return sorted feasible ion masses for one (neutral formula, precursor type)."""
    return _subformula_masses(formula, prec_type)


def peak_feasibility(peaks, formula, prec_type, inst_type):
    """Return (mzs, intensities, feasible mask) for one spectrum.

    Mirrors the searchsorted acceptance test in ``filter_subformula_peaks`` so the probe and
    the production filter agree peak for peak.
    """
    mzs = np.asarray([float(p[0]) for p in peaks], dtype=np.float64)
    ints = np.asarray([float(p[1]) for p in peaks], dtype=np.float64)
    if mzs.size == 0:
        return mzs, ints, np.zeros(0, dtype=bool)

    candidates = feasible_masses(formula, prec_type)
    if candidates.size == 0:
        return mzs, ints, np.zeros(mzs.size, dtype=bool)

    tol_cfg = INSTRUMENT_TOLERANCES[inst_type]
    half_width = tol_cfg["relative"] * np.maximum(mzs, tol_cfg["min_mz"])
    insert = np.searchsorted(candidates, mzs)
    feasible = np.zeros(mzs.size, dtype=bool)
    for offset in (-1, 0):
        idx = np.clip(insert + offset, 0, candidates.size - 1)
        feasible |= np.abs(mzs - candidates[idx]) <= half_width
    return mzs, ints, feasible


NEUTRON_DELTA = 1.003355  # 13C-12C, the spacing of an isotopologue ladder
MAX_ISOTOPE_K = 3


def precursor_classes(mzs: np.ndarray, prec_mz: float, half_width: np.ndarray) -> dict:
    """Classify peaks relative to the precursor m/z.

    Returns boolean masks for the precursor peak itself, its isotopologue ladder, any other
    above-precursor peak, and everything below. The ladder is separated because a monoisotopic
    candidate set cannot match it however good the enumeration is, so it is real signal that
    looks like an unexplainable peak.
    """
    if not np.isfinite(prec_mz):
        below = np.ones(mzs.size, dtype=bool)
        empty = np.zeros(mzs.size, dtype=bool)
        return {"at_prec": empty, "iso_prec": empty, "above_other": empty, "below": below}

    at_prec = np.abs(mzs - prec_mz) <= half_width
    iso_prec = np.zeros(mzs.size, dtype=bool)
    for k in range(1, MAX_ISOTOPE_K + 1):
        iso_prec |= np.abs(mzs - (prec_mz + k * NEUTRON_DELTA)) <= half_width
    iso_prec &= ~at_prec
    above = (mzs > prec_mz) & ~at_prec
    return {
        "at_prec": at_prec,
        "iso_prec": iso_prec & above,
        "above_other": above & ~iso_prec,
        "below": ~above & ~at_prec,
    }


def isotopologue_of_feasible(
    mzs: np.ndarray, feasible: np.ndarray, half_width: np.ndarray
) -> np.ndarray:
    """Flag infeasible peaks sitting one or two neutron masses above a feasible peak.

    This is a mass-only test, so it is an upper bound on the isotopologue population: it does
    not check that the intensity ratio is consistent with the carbon count. It is still far
    more specific than "no subformula", because an arbitrary contaminant has no reason to sit
    exactly one neutron above an explainable fragment of the same molecule.
    """
    flagged = np.zeros(mzs.size, dtype=bool)
    if not feasible.any():
        return flagged
    parents = np.sort(mzs[feasible])
    for k in (1, 2):
        shifted = mzs - k * NEUTRON_DELTA
        insert = np.searchsorted(parents, shifted)
        for offset in (-1, 0):
            idx = np.clip(insert + offset, 0, parents.size - 1)
            flagged |= np.abs(shifted - parents[idx]) <= half_width
    return flagged & ~feasible


def donor_formulas(spec_df: pd.DataFrame, formulas: pd.Series, seed: int) -> list[str]:
    """Return one wrong-but-plausible formula per spectrum.

    Donors are drawn within precursor-mass deciles so a donor admits a subformula grid of
    comparable size to the true molecule's. Without that banding a light donor would reject
    almost every peak and overstate the test's power.
    """
    rng = np.random.default_rng(seed)
    prec = pd.to_numeric(spec_df["prec_mz"], errors="coerce").to_numpy()
    band = pd.qcut(pd.Series(prec), 10, labels=False, duplicates="drop").to_numpy()
    mol_ids = spec_df["mol_id"].to_numpy()
    donors = [None] * len(spec_df)
    for band_id in np.unique(band[~pd.isna(band)]):
        members = np.flatnonzero(band == band_id)
        pool = mol_ids[members]
        for slot in members:
            choice = mol_ids[slot]
            for _ in range(8):  # resample rather than accept the molecule's own formula
                candidate = pool[rng.integers(len(pool))]
                if candidate != mol_ids[slot]:
                    choice = candidate
                    break
            donors[slot] = formulas.at[choice]
    fallback = formulas.iloc[0]
    return [d if d is not None else fallback for d in donors]


def weighted_share(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Return per-group intensity-weighted infeasible share and spectrum counts."""
    grouped = frame.groupby(column, observed=True)
    out = pd.DataFrame(
        {
            "n_spec": grouped.size(),
            "infeasible_int": grouped["infeasible_int"].sum() / grouped["total_int"].sum(),
            "infeasible_peaks": grouped["n_infeasible"].sum() / grouped["n_peaks"].sum(),
            "median_per_spec": grouped["infeasible_frac"].median(),
        }
    )
    return out.sort_values("n_spec", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template_fp", default="config")
    parser.add_argument("--custom_fp", required=True, help="campaign config whose target to probe")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument(
        "--measured_wrecall",
        type=float,
        default=None,
        help="a run's candidate coverage on this split; splits the residual when given",
    )
    parser.add_argument("--n_sample", type=int, default=0, help="0 uses every spectrum")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--null_control",
        action="store_true",
        help="also score every spectrum against a wrong donor formula of similar precursor mass",
    )
    parser.add_argument("--out_csv", default=None, help="optional per-spectrum dump")
    args = parser.parse_args()

    # 1. Resolve the run's own configuration so the probed target is the trained target.
    config = load_config(args.template_fp, args.custom_fp, [])
    spec_params = config["spec_params"]
    print(f"config          : {args.custom_fp}")
    print(f"split           : {args.split} of {config['split_dp']}")
    print(f"merge           : {spec_params['merge']} / {spec_params['merge_method']}")
    print(f"subformula filt : {spec_params['filter_subformula_peaks']} (probe measures, never filters)")
    print(f"adducts         : {spec_params['prec_types']}")
    print(f"instruments     : {spec_params['inst_types']}  modes: {spec_params['frag_modes']}")

    # 2. Build the merged target exactly as the dataset does.
    spec_df, mol_df, _, _, _, _ = BaseDataset._setup_dfs(
        config["spec_fp"],
        config["mol_fp"],
        config["split_dp"],
        [args.split],
        config["subsample_params"],
        spec_params,
    )
    if args.n_sample and args.n_sample < len(spec_df):
        spec_df = spec_df.sample(n=args.n_sample, random_state=args.seed)
    formulas = mol_df.set_index("mol_id")["formula"]
    print(f"\nspectra probed  : {len(spec_df):,}  molecules: {spec_df['mol_id'].nunique():,}")

    # 3. Score every peak for subformula feasibility.
    donors = donor_formulas(spec_df, formulas, args.seed) if args.null_control else None
    records = []
    for position, row in enumerate(spec_df.itertuples(index=False)):
        mzs, ints, feasible = peak_feasibility(
            row.peaks, formulas.at[row.mol_id], row.prec_type, row.inst_type
        )
        total = ints.sum()
        if total <= 0.0:
            continue
        infeasible_int = ints[~feasible].sum()
        prec_mz = float(getattr(row, "prec_mz", np.nan))
        tol_cfg = INSTRUMENT_TOLERANCES[row.inst_type]
        half_width = tol_cfg["relative"] * np.maximum(mzs, tol_cfg["min_mz"])
        classes = precursor_classes(mzs, prec_mz, half_width)
        above_prec = classes["iso_prec"] | classes["above_other"]
        iso_child = isotopologue_of_feasible(mzs, feasible, half_width)
        # The precursor's own M+1 satisfies both tests, so take the union, never the sum.
        iso_any = (iso_child | classes["iso_prec"]) & ~feasible
        records.append(
            {
                "mol_id": row.mol_id,
                "prec_type": row.prec_type,
                "inst_type": row.inst_type,
                "frag_mode": row.frag_mode,
                "n_peaks": mzs.size,
                "n_infeasible": int((~feasible).sum()),
                "total_int": total,
                "infeasible_int": infeasible_int,
                "infeasible_frac": infeasible_int / total,
                "infeasible_above_prec_int": ints[(~feasible) & above_prec].sum(),
                "feasible_int": ints[feasible].sum(),
                "at_prec_int": ints[classes["at_prec"]].sum(),
                "iso_prec_int": ints[classes["iso_prec"]].sum(),
                "above_other_int": ints[classes["above_other"]].sum(),
                "iso_prec_infeasible_int": ints[classes["iso_prec"] & ~feasible].sum(),
                "above_other_infeasible_int": ints[classes["above_other"] & ~feasible].sum(),
                "iso_of_feasible_int": ints[iso_child].sum(),
                "iso_of_feasible_below_int": ints[iso_child & classes["below"]].sum(),
                "n_iso_of_feasible": int(iso_child.sum()),
                "iso_any_int": ints[iso_any].sum(),
                "n_iso_any": int(iso_any.sum()),
                "contaminant_int": ints[(~feasible) & ~iso_any].sum(),
                "n_contaminant": int(((~feasible) & ~iso_any).sum()),
                "n_above_other": int(classes["above_other"].sum()),
                "n_iso_prec": int(classes["iso_prec"].sum()),
                "max_infeasible_rel": (ints[~feasible].max() / ints.max()) if (~feasible).any() else 0.0,
            }
        )
        if donors is not None:
            _, _, donor_feasible = peak_feasibility(
                row.peaks, donors[position], row.prec_type, row.inst_type
            )
            records[-1]["donor_infeasible_int"] = ints[~donor_feasible].sum()
            records[-1]["donor_n_infeasible"] = int((~donor_feasible).sum())
    frame = pd.DataFrame(records)

    # 4. Headline decomposition.
    total_int = frame["total_int"].sum()
    infeasible = frame["infeasible_int"].sum() / total_int
    feasible = 1.0 - infeasible
    print("\n=== 1. Intensity-weighted feasibility of the merged target ===")
    print(f"  feasible subformula      : {feasible:.4f}")
    print(f"  NO feasible subformula   : {infeasible:.4f}   <- escape population")
    print(f"  peaks with no subformula : {frame['n_infeasible'].sum() / frame['n_peaks'].sum():.4f} "
          f"({frame['n_infeasible'].sum():,} of {frame['n_peaks'].sum():,})")

    if args.measured_wrecall is not None:
        coverage_gap = feasible - args.measured_wrecall
        residual = 1.0 - args.measured_wrecall
        print("\n=== 2. Splitting the unexplained residual ===")
        print(f"  measured candidate coverage (wrecall) : {args.measured_wrecall:.4f}")
        print(f"  unexplained residual                  : {residual:.4f}")
        print(f"    -> not from this molecule (escape)  : {infeasible:.4f} "
              f"({infeasible / residual:5.1%} of residual)")
        print(f"    -> feasible but unreached (coverage): {coverage_gap:.4f} "
              f"({coverage_gap / residual:5.1%} of residual)")

    if "donor_infeasible_int" in frame.columns:
        donor = frame["donor_infeasible_int"].sum() / total_int
        print("\n=== 2b. Null control: same peaks, wrong molecule ===")
        print(f"  infeasible under the TRUE formula  : {infeasible:.4f}")
        print(f"  infeasible under a DONOR formula   : {donor:.4f} "
              f"(matched precursor-mass band)")
        print(f"  discrimination (donor / true)      : {donor / infeasible:.2f}x")
        print(f"  donor infeasible peak fraction     : "
              f"{frame['donor_n_infeasible'].sum() / frame['n_peaks'].sum():.4f}")

    # 5. Concentration decides per-spectrum gate versus per-peak responsibility.
    print("\n=== 3. How concentrated is the infeasible mass? ===")
    quantiles = frame["infeasible_frac"].quantile([0.5, 0.75, 0.9, 0.95, 0.99])
    for q, v in quantiles.items():
        print(f"  per-spectrum infeasible fraction p{int(q * 100):<2d} : {v:.4f}")
    ordered = frame.sort_values("infeasible_int", ascending=False)
    cumulative = ordered["infeasible_int"].cumsum() / ordered["infeasible_int"].sum()
    for share in (0.1, 0.25, 0.5):
        n_spec = int(np.searchsorted(cumulative.to_numpy(), share) + 1)
        print(f"  {share:4.0%} of all infeasible mass sits in {n_spec:,} spectra "
              f"({n_spec / len(frame):5.2%} of the split)")
    for threshold in (0.05, 0.10, 0.25, 0.50):
        hit = (frame["infeasible_frac"] > threshold).mean()
        print(f"  spectra with >{threshold:4.0%} infeasible intensity : {hit:6.2%}")
    print(f"  spectra whose BASE PEAK is infeasible       : "
          f"{(frame['max_infeasible_rel'] >= 1.0).mean():6.2%}")

    # 6. Cheap structural subsets and stratification.
    print("\n=== 4. Everything above the precursor m/z ===")
    at_prec = frame["at_prec_int"].sum() / total_int
    iso = frame["iso_prec_int"].sum() / total_int
    other = frame["above_other_int"].sum() / total_int
    print(f"  precursor peak itself                : {at_prec:.4f}")
    print(f"  precursor isotopologue ladder (M+1..3): {iso:.4f}   REAL SIGNAL, monoisotopic "
          f"candidates cannot match it")
    print(f"  any other peak above the precursor   : {other:.4f}   co-isolation / contaminant / "
          f"multiply charged")
    print(f"  total above precursor                : {iso + other:.4f}")
    iso_infeasible = frame["iso_prec_infeasible_int"].sum() / total_int
    other_infeasible = frame["above_other_infeasible_int"].sum() / total_int
    print(f"\n  of the escape population ({infeasible:.4f}):")
    print(f"    isotopologue ladder                : {iso_infeasible:.4f} "
          f"({iso_infeasible / infeasible:5.1%})")
    print(f"    other above-precursor              : {other_infeasible:.4f} "
          f"({other_infeasible / infeasible:5.1%})")
    rest = infeasible - iso_infeasible - other_infeasible
    print(f"    below the precursor                : {rest:.4f} ({rest / infeasible:5.1%})")
    print(f"\n  mean above-precursor non-isotope peaks per spectrum : "
          f"{frame['n_above_other'].mean():.2f}")
    print(f"  spectra with any such peak                          : "
          f"{(frame['n_above_other'] > 0).mean():6.2%}")

    print("\n=== 4b. Is the escape population isotopologues rather than contaminants? ===")
    iso_child_total = frame["iso_of_feasible_int"].sum() / total_int
    iso_child_below = frame["iso_of_feasible_below_int"].sum() / total_int
    print(f"  infeasible AND one/two neutrons above a feasible peak : {iso_child_total:.4f} "
          f"({iso_child_total / infeasible:5.1%} of the escape population)")
    print(f"    of which below the precursor                        : {iso_child_below:.4f}")
    print(f"  infeasible peaks flagged as isotopologues             : "
          f"{frame['n_iso_of_feasible'].sum():,} of {frame['n_infeasible'].sum():,} "
          f"({frame['n_iso_of_feasible'].sum() / frame['n_infeasible'].sum():.1%})")
    iso_any_total = frame["iso_any_int"].sum() / total_int
    residue = frame["contaminant_int"].sum() / total_int
    print(f"  union of both isotopologue tests                      : {iso_any_total:.4f} "
          f"({iso_any_total / infeasible:5.1%} of the escape population)")
    print(f"    (the precursor's own M+1 satisfies both tests, so the union is taken; "
          f"the two overlap by {iso_child_total + frame['iso_prec_int'].sum() / total_int - iso_any_total:.4f})")
    print(f"\n  escape population NOT explained by any isotopologue   : {residue:.4f} "
          f"({residue / infeasible:5.1%} of the escape population, "
          f"{residue:.4f} of all target intensity)")
    print(f"  peaks in that residue                                 : "
          f"{frame['n_contaminant'].sum():,} of {frame['n_infeasible'].sum():,}")
    print("  ^ this is the only part a contaminant/noise escape would address")

    print("\n=== 5. Stratification (intensity-weighted infeasible share) ===")
    for column in ("prec_type", "inst_type", "frag_mode"):
        print(f"\n-- by {column}")
        print(weighted_share(frame, column).to_string(float_format=lambda x: f"{x:9.4f}"))

    if args.out_csv:
        frame.to_csv(args.out_csv, index=False)
        print(f"\nper-spectrum dump: {args.out_csv}")


if __name__ == "__main__":
    main()
