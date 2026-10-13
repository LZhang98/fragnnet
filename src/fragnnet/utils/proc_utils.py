import math

import numpy as np
import pandas as pd

from fragnnet.utils.clean_spectra_utils import merge_peaks
from fragnnet.utils.formula_utils import MAX_NUM_EDGES, MAX_NUM_NODES, parse_formula


def element_filter(formula, elements):
    try:
        element_counts = parse_formula(formula)
    except ValueError:
        return False
    return all(v <= 0 or k in elements for k, v in element_counts.items())


def filter_spec_mol(
    spec_df,
    mol_df,
    elements=None,
    dsets=None,
    max_peak_mz=None,
    max_prec_mz=None,
    min_prec_mz=None,
    prec_types: list | None = None,
    num_entries=-1,
    inst_types: list | None = None,
    frag_modes: list | None = None,
    ion_modes: list | None = None,
    ces: str | None = None,
    spec_type: str | None = None,
    max_heavy_atom: int | None = None,
    max_bond: int | None = None,
    ce_types: list | None = None,
):
    """
    The purpose of this filtering is to prevent errors
    """
    masks = []

    ## spectrum criteria (TODO remove this)
    # dataset mask
    if dsets is not None:
        print(f">> dsets filter: {dsets}")
        print(">> dsets in data:", spec_df["dset"].unique())
        dset_mask = spec_df["dset"].isin(dsets)
        print(f">> dsets filter mean: {dset_mask.mean()}")
        masks.append(dset_mask)
    else:
        print(">> no dset filter")
        print(">> dsets in data:", spec_df["dset"].unique())

    # instrument type
    if inst_types is not None:
        print(f">> inst type: {inst_types}")
        print(">> inst type in data:", spec_df["inst_type"].unique())
        inst_type_mask = spec_df["inst_type"].isin(inst_types)
        print(f">> inst type filter mean: {inst_type_mask.mean()}")
        masks.append(inst_type_mask)
    else:
        print(">> no inst type filter")
        print(">> inst type in data:", spec_df["inst_type"].unique())

    # frag mode
    if frag_modes is not None:
        print(f">> frag mode filter: {frag_modes}")
        print(">> frag mode in data:", spec_df["frag_mode"].unique())
        frag_mode_mask = spec_df["frag_mode"].isin(frag_modes)
        print(f">> frag mode filter mean: {frag_mode_mask.mean()}")
        masks.append(frag_mode_mask)
    else:
        print(">> no frag mode filter")
        if "frag_mode" in spec_df:
            print(">> frag mode in data:", spec_df["frag_mode"].unique())

    # ce filter
    assert ces in ["ace", "nce", "nce_or_ace", None], ces
    if ces == "ace":
        print(">> require ace")
        ace_mask = ~spec_df["ace"].isna()
        print(f">> ace filter: {ace_mask.mean()}")
        masks.append(ace_mask)
    elif ces == "nce":
        print(">> require nce")
        nce_mask = ~spec_df["nce"].isna()
        print(f">> nce filter: {nce_mask.mean()}")
        masks.append(nce_mask)
    elif ces == "nce_or_ace":
        print(">> require ace or nce")
        mask = ~spec_df["nce"].isna() | ~spec_df["ace"].isna()
        print(f">> ace_or_nce filter: {mask.mean()}")
        masks.append(mask)
    else:
        print(f">> ces filter set to {ces},  no ce filter applied")

    # ce_types
    # print(ce_types)
    ce_types = ce_types if ce_types is not None else []
    if ce_types:
        print(f">> ce_types filter: {ce_types}")
        allowed = {"stepped", "ramped", "single", "none"}
        unknown = set(ce_types) - allowed
        if unknown:
            raise ValueError(f"unknown ce_types requested: {sorted(unknown)}")
        ce_mask = spec_df["ce_type"].isin(ce_types)
        print(f">> ce_type filter mean: {ce_mask.mean()}")
        masks.append(ce_mask)
    else:
        print(">> no ce_type filter")

    # ion mode
    if ion_modes is not None:
        print(f">> ion mode filter: {ion_modes}")
        print(">> ion mode in data:", spec_df["ion_mode"].unique())
        ion_mode_mask = spec_df["ion_mode"].isin(ion_modes)
        print(f">> ion mode filter mean: {ion_mode_mask.mean()}")
        masks.append(ion_mode_mask)
    else:
        print(">> no ion mode filter")
        print(">> ion mode in data:", spec_df["ion_mode"].unique())

    # precursor type
    if prec_types is not None:
        print(f">> prec type filter: {prec_types}")
        print(">> prec type in data: ", spec_df["prec_type"].unique())
        prec_type_mask = spec_df["prec_type"].isin(prec_types)
        print(f">> prec type filter mean: {prec_type_mask.mean()}")
        masks.append(prec_type_mask)
    else:
        print(">> no prec type filter")
        print(">> prec type in data: ", spec_df["prec_type"].unique())

    # resolution
    # res_mask = spec_df["res"].isin([1,2,3,4,5,6,7])
    # print(f">> res: {res_mask.mean()}")
    # masks.append(res_mask)

    # spectrum type
    if spec_type is not None:
        print(f">> spec type filter: {spec_type}")
        print(">> spec type in data: ", spec_df["spec_type"].unique())
        spec_type_mask = spec_df["spec_type"] == spec_type
        print(f">> spec type: {spec_type_mask.mean()}")
        masks.append(spec_type_mask)
    else:
        print(">> no spec type filter")
        print(">> spec type in data: ", spec_df["spec_type"].unique())

    # precursor mz
    if "prec_mz" in spec_df:
        prec_mz_mask = ~spec_df["prec_mz"].isna()
        print(f">> prec mz: {prec_mz_mask.mean()}")
        masks.append(prec_mz_mask)

    # max prec mz
    if max_prec_mz is not None:
        print(f">> max_prec_mz value: {max_prec_mz}")
        max_prec_mz_mask = spec_df["prec_mz"] <= max_prec_mz
        print(f">> max prec mz: {max_prec_mz_mask.mean()}")
        masks.append(max_prec_mz_mask)

    # min prec mz
    if min_prec_mz is not None:
        print(f">> min_prec_mz value: {min_prec_mz}")
        min_prec_mz_mask = spec_df["prec_mz"] >= min_prec_mz
        print(f">> min prec mz: {min_prec_mz_mask.mean()}")
        masks.append(min_prec_mz_mask)

    # max peak mz
    if max_peak_mz is not None:

        def get_max_mz(peaks):
            return max([peak[0] for peak in peaks])

        print(f">> max_peak_mz value: {max_peak_mz}")
        max_peak_mz_mask = spec_df["peaks"].apply(get_max_mz) <= max_peak_mz
        print(f">> max peak mz: {max_peak_mz_mask.mean()}")
        masks.append(max_peak_mz_mask)

    ## molecule criteria
    # single molecule
    single_mol_ids = mol_df[mol_df["single_mol"]]["mol_id"]
    single_mol_mask = spec_df["mol_id"].isin(single_mol_ids)
    print(f">> single mol: {single_mol_mask.mean()}")
    masks.append(single_mol_mask)
    # neutral or pre-formed +1 cation ([M]+ support: quaternary ammonium, sulfonium, etc.)
    neutral_ids = mol_df[mol_df["charge"].isin([0, 1])]["mol_id"]
    neutral_mask = spec_df["mol_id"].isin(neutral_ids)
    print(f">> neutral or [M]+: {neutral_mask.mean()}")
    masks.append(neutral_mask)
    # Cross-check: charge=+1 molecules are only valid with [M]+ spectra.
    # A charge=+1 molecule paired with [M+H]+ would apply +H to already-charged
    # fragment formulas, producing wrong m/z values.
    charged_mol_ids = set(mol_df[mol_df["charge"] == 1]["mol_id"].tolist())
    if charged_mol_ids:
        bad_adduct_mask = spec_df["mol_id"].isin(charged_mol_ids) & (spec_df["prec_type"] != "[M]+")
        n_bad = int(bad_adduct_mask.sum())
        if n_bad > 0:
            print(
                f">> charge cross-check: dropping {n_bad} spectra of charge=+1 molecules "
                f"with prec_type != '[M]+'"
            )
        masks.append(~bad_adduct_mask)
    # element composition
    if elements is not None:
        element_ids = mol_df[
            mol_df["formula"].apply(lambda formula: element_filter(formula, elements))
        ]["mol_id"]
        element_mask = spec_df["mol_id"].isin(element_ids)
        print(f">> element: {element_mask.mean()}")
        masks.append(element_mask)

    # atom count
    max_heavy_atom = max_heavy_atom if max_heavy_atom is not None else MAX_NUM_NODES
    print(f">> max atoms allowed: {max_heavy_atom}")
    atom_count_ids = mol_df[mol_df["num_atoms"] <= max_heavy_atom]["mol_id"]
    atom_count_mask = spec_df["mol_id"].isin(atom_count_ids)
    print(f">> atom count: {atom_count_mask.mean()}")
    masks.append(atom_count_mask)

    max_bond = max_bond if max_bond is not None else MAX_NUM_EDGES
    print(f">> max bonds allowed: {max_bond}")
    # bond count
    bond_count_ids = mol_df[mol_df["num_bonds"] <= max_bond]["mol_id"]
    bond_count_mask = spec_df["mol_id"].isin(bond_count_ids)
    print(f">> bond count: {bond_count_mask.mean()}")
    masks.append(bond_count_mask)

    # radical stats
    non_radical_ids = mol_df[mol_df["num_radicals"] == 0]["mol_id"]
    non_radical_mask = spec_df["mol_id"].isin(non_radical_ids)
    print(f">> non radical: {non_radical_mask.mean()}")
    masks.append(non_radical_mask)
    # put them together
    all_mask = masks[0]
    for mask in masks:
        all_mask = all_mask & mask
    if np.sum(all_mask) == 0:
        raise ValueError("select removed all items")
    print(f">> everything: {all_mask.mean()}")
    spec_df = spec_df[all_mask].reset_index(drop=True)
    # sample
    if num_entries > 0:
        spec_df = spec_df.sample(n=num_entries, replace=False, random_state=420).reset_index(
            drop=True
        )
    # only keep molecules that are in the spectra
    mol_df = mol_df[mol_df["mol_id"].isin(spec_df["mol_id"])].reset_index(drop=True)
    return spec_df, mol_df


def _normalize_peak_list(peaks: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """L1-normalize one source spectrum before consensus aggregation."""
    total_intensity = sum(float(intensity) for _, intensity in peaks)
    if total_intensity <= 0.0:
        raise ValueError("Cannot normalize a spectrum with nonpositive total intensity")
    return [(float(mz), float(intensity) / total_intensity) for mz, intensity in peaks]


def _ce_group_key(value) -> tuple[float, ...]:
    """Return a stable key for scalar, stepped, or ramped collision energies."""
    values = value if isinstance(value, (list, tuple, np.ndarray)) else [value]
    return tuple(-1.0 if pd.isna(item) else round(float(item), 6) for item in values)


def _merge_peak_lists_with_tolerance(
    peak_lists: list[list[tuple[float, float]]],
    *,
    tolerance_ppm: float,
    tolerance_min_mz: float,
    require_common_tolerance_overlap: bool,
) -> list[tuple[float, float]]:
    """Tolerance-cluster peak lists and average their intensities."""
    if not peak_lists:
        raise ValueError("At least one peak list is required for consensus merging")
    merged = merge_peaks(
        [peak for peaks in peak_lists for peak in peaks],
        tolerance_ppm=tolerance_ppm,
        tolerance_min_mz=tolerance_min_mz,
        require_common_tolerance_overlap=require_common_tolerance_overlap,
    )
    divisor = float(len(peak_lists))
    return [(mz, intensity / divisor) for mz, intensity in merged]


def _consensus_group_peaks(
    group_df,
    *,
    ce_key: str | None,
    tolerance_by_inst: dict,
    require_common_tolerance_overlap: bool,
    normalize_each_spectrum: bool,
    equal_ce_weight: bool,
) -> list[tuple[float, float]]:
    """Build an equal-spectrum or equal-CE tolerance-aware group consensus."""
    inst_types = group_df["inst_type"].drop_duplicates().tolist()
    if len(inst_types) != 1:
        raise ValueError(f"Merged group must have one instrument type, got {inst_types}")
    inst_type = inst_types[0]
    if inst_type not in tolerance_by_inst:
        raise ValueError(f"No merge tolerance configured for instrument type {inst_type!r}")
    tolerance = tolerance_by_inst[inst_type]
    tolerance_ppm = float(tolerance["rel"]) * 1e6
    tolerance_min_mz = float(tolerance["min_mz"])

    # 1. Normalize source spectra so acquisition-specific intensity scales do not
    # control their contribution to the merged target.
    source_peak_lists = [
        _normalize_peak_list(peaks) if normalize_each_spectrum else list(peaks)
        for peaks in group_df["peaks"]
    ]
    if not equal_ce_weight:
        return _merge_peak_lists_with_tolerance(
            source_peak_lists,
            tolerance_ppm=tolerance_ppm,
            tolerance_min_mz=tolerance_min_mz,
            require_common_tolerance_overlap=require_common_tolerance_overlap,
        )
    if ce_key is None:
        raise ValueError("Equal-CE consensus merging requires nce or ace spectrum context")

    # 2. Average replicate spectra within each distinct CE schedule.
    peak_lists_by_ce: dict[tuple[float, ...], list[list[tuple[float, float]]]] = {}
    for ce_value, peaks in zip(group_df[ce_key], source_peak_lists, strict=True):
        peak_lists_by_ce.setdefault(_ce_group_key(ce_value), []).append(peaks)
    ce_consensus_peak_lists = [
        _merge_peak_lists_with_tolerance(
            peak_lists,
            tolerance_ppm=tolerance_ppm,
            tolerance_min_mz=tolerance_min_mz,
            require_common_tolerance_overlap=require_common_tolerance_overlap,
        )
        for peak_lists in peak_lists_by_ce.values()
    ]

    # 3. Give each distinct CE schedule equal weight in the final target.
    return _merge_peak_lists_with_tolerance(
        ce_consensus_peak_lists,
        tolerance_ppm=tolerance_ppm,
        tolerance_min_mz=tolerance_min_mz,
        require_common_tolerance_overlap=require_common_tolerance_overlap,
    )


def _weighted_merge_ce_context(spec_df: pd.DataFrame, ce_key: str) -> pd.DataFrame:
    """Preserve CE readings with equal schedule, replicate, and within-row weights.

    Returns one row per group with sorted, coalesced NCE/ACE readings and matching
    weights. For G schedules, R replicates, and L readings, each reading receives
    weight 1/(G*R*L). Spectral targets and their grouping are not modified.
    """
    records = []
    source = spec_df.assign(_merge_ce_key=spec_df[ce_key].map(_ce_group_key))
    for group_id, group in source.groupby("group_id", sort=True):
        schedules = list(group.groupby("_merge_ce_key", sort=True))
        record = {"group_id": group_id}
        for signal in ("nce", "ace"):
            contributions: dict[float, list[float]] = {}
            for _, schedule in schedules:
                for value in schedule[signal]:
                    readings = _ce_group_key(value)
                    if not readings:
                        raise ValueError("Weighted CE aggregation requires nonempty CE readings")
                    weight = 1.0 / (len(schedules) * len(schedule) * len(readings))
                    for reading in readings:
                        contributions.setdefault(reading, []).append(weight)
            readings = sorted(contributions)
            record[signal] = readings
            record[f"{signal}_weights"] = [
                math.fsum(sorted(contributions[reading])) for reading in readings
            ]
        records.append(record)
    return pd.DataFrame.from_records(
        records, columns=["group_id", "nce", "nce_weights", "ace", "ace_weights"]
    )


def merge_spec_df(
    spec_df,
    renormalize=False,
    sum_ints=True,
    keep_ces=False,
    *,
    method="exact_sum",
    ce_key=None,
    tolerance_by_inst=None,
    require_common_tolerance_overlap=True,
    normalize_each_spectrum=False,
    equal_ce_weight=False,
    ce_aggregation="legacy",
):
    """Merge spectra by exact m/z or build a tolerance-aware consensus.

    ``ce_aggregation='legacy'`` preserves historical CE context. ``'weighted'``
    retains all CE observations and adds ``nce_weights``/``ace_weights`` with
    equal schedule/replicate weights; it requires equal-CE consensus and does not
    alter spectral target aggregation.
    """
    if method not in {"exact_sum", "tolerance_consensus"}:
        raise ValueError(f"Unknown spectrum merge method: {method!r}")
    if ce_aggregation not in {"legacy", "weighted"}:
        raise ValueError(f"Unknown CE aggregation: {ce_aggregation!r}")
    if ce_aggregation == "weighted" and not (
        keep_ces
        and method == "tolerance_consensus"
        and equal_ce_weight
        and ce_key in {"nce", "ace"}
    ):
        raise ValueError("Weighted CE aggregation requires retained CE and equal-CE consensus")

    # Legacy exact-m/z behavior remains the default for checkpoint reproducibility.
    def m_peaks_func(peakses):
        return merge_sparse_specs(*peakses, renormalize=renormalize, sum_ints=sum_ints)

    if method == "exact_sum":
        m_spec_peaks_df = (
            spec_df[["group_id", "peaks"]]
            .groupby("group_id")
            .agg({"peaks": m_peaks_func})
            .reset_index()
        )
    else:
        if tolerance_by_inst is None:
            raise ValueError("tolerance_consensus requires tolerance_by_inst")
        merged_groups = []
        for group_id, group_df in spec_df.groupby("group_id", sort=False):
            peaks = _consensus_group_peaks(
                group_df,
                ce_key=ce_key,
                tolerance_by_inst=tolerance_by_inst,
                require_common_tolerance_overlap=require_common_tolerance_overlap,
                normalize_each_spectrum=normalize_each_spectrum,
                equal_ce_weight=equal_ce_weight,
            )
            if renormalize:
                peaks = _normalize_peak_list(peaks)
            merged_groups.append({"group_id": group_id, "peaks": peaks})
        m_spec_peaks_df = pd.DataFrame.from_records(merged_groups)

    m_spec_meta_df = spec_df.drop(columns=["peaks", "spec_id"]).drop_duplicates(subset=["group_id"])
    m_spec_df = m_spec_peaks_df.merge(m_spec_meta_df, on=["group_id"], how="inner")
    if keep_ces:
        # After _fill_ce_list (Phase 2 in base_dataset), each cell is a list.
        # When called directly (e.g. tests) cells may still be scalars.
        # Flatten either way so th.tensor() gets a uniform 1-D list.
        def _flatten(x):
            result = []
            for item in x:
                if isinstance(item, list):
                    result.extend(item)
                else:
                    result.append(item)
            return result

        if ce_aggregation == "weighted":
            m_spec_ce_df = _weighted_merge_ce_context(spec_df, ce_key)
        else:
            ce_source_df = spec_df
            if method == "tolerance_consensus" and equal_ce_weight:
                ce_source_df = spec_df.assign(
                    _merge_ce_key=spec_df[ce_key].map(_ce_group_key)
                ).drop_duplicates(subset=["group_id", "_merge_ce_key"])
            m_spec_ce_df = (
                ce_source_df[["group_id", "nce", "ace"]]
                .groupby("group_id")
                .agg({"nce": _flatten, "ace": _flatten})
                .reset_index()
            )
        m_spec_df = m_spec_df.drop(columns=["nce", "ace"])
        m_spec_df = m_spec_df.merge(m_spec_ce_df, on=["group_id"], how="inner")
    else:
        m_spec_df.loc[:, "nce"] = np.nan
        m_spec_df.loc[:, "ace"] = np.nan
    return m_spec_df


def merge_sparse_specs(
    *peakses, renormalize: bool = False, sum_ints: bool = True
) -> list[tuple[float, float]]:
    """
    this will result in peaks that are really close in mass (<5ppm)
    they are probably the same peak, but our model can handle this type of ambiguity
    for now, let's keep them unmerged

    Args:
        renormalize (bool, optional): flag to get renormalize spectra. Defaults to False.
        sum_ints (bool, optional): flag for sum intensities within the bin, else take max

    Returns:
        _type_: _description_
    """
    merged_peaks = {}
    # total_intensity = 0.
    for peaks in peakses:
        for mz, intensity in peaks:
            if mz in merged_peaks:
                if sum_ints:
                    merged_peaks[mz] += intensity
                else:
                    merged_peaks[mz] = max(merged_peaks[mz], intensity)
            else:
                merged_peaks[mz] = intensity
    merged_peaks = sorted(merged_peaks.items(), key=lambda x: x[0])
    if renormalize:
        total_intensity = sum([intensity for mz, intensity in merged_peaks])
        merged_peaks = [(mz, intensity / total_intensity) for mz, intensity in merged_peaks]
    return merged_peaks
