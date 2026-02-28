import numpy as np
import pandas as pd
import pytest

from fragnnet.utils.proc_utils import element_filter, filter_spec_mol, merge_spec_df


def make_spec_and_mol_frames():
    spec_df = pd.DataFrame(
        [
            {
                "dset": "nist",
                "inst_type": "orbitrap",
                "frag_mode": "hcd",
                "ce_type": "stepped",
                "nce": 20.0,
                "ace": np.nan,
                "ion_mode": "positive",
                "prec_type": "[M+H]+",
                "spec_type": "centroid",
                "prec_mz": 150.0,
                "peaks": [(10.0, 0.5), (20.0, 0.5)],
                "mol_id": 1,
                "spec_id": 101,
                "group_id": 1,
            },
            {
                "dset": "nist",
                "inst_type": "orbitrap",
                "frag_mode": "hcd",
                "ce_type": "single",
                "nce": np.nan,
                "ace": np.nan,
                "ion_mode": "positive",
                "prec_type": "[M+H]+",
                "spec_type": "centroid",
                "prec_mz": 220.0,
                "peaks": [(10.0, 1.0)],
                "mol_id": 2,
                "spec_id": 102,
                "group_id": 1,
            },
            {
                "dset": "nist",
                "inst_type": "orbitrap",
                "frag_mode": "cid",
                "ce_type": "single",
                "nce": 25.0,
                "ace": np.nan,
                "ion_mode": "positive",
                "prec_type": "[M+Na]+",
                "spec_type": "centroid",
                "prec_mz": 120.0,
                "peaks": [(15.0, 0.7)],
                "mol_id": 3,
                "spec_id": 103,
                "group_id": 2,
            },
        ]
    )
    mol_df = pd.DataFrame(
        [
            {
                "mol_id": 1,
                "single_mol": True,
                "charge": 0,
                "formula": "C2H6O",
                "num_atoms": 3,
                "num_bonds": 2,
                "num_radicals": 0,
            },
            {
                "mol_id": 2,
                "single_mol": True,
                "charge": 0,
                "formula": "C6H6",
                "num_atoms": 6,
                "num_bonds": 6,
                "num_radicals": 0,
            },
            {
                "mol_id": 3,
                "single_mol": True,
                "charge": 0,
                "formula": "C6H5Cl",
                "num_atoms": 7,
                "num_bonds": 6,
                "num_radicals": 0,
            },
        ]
    )
    return spec_df, mol_df


def test_element_filter_behaviors():
    assert element_filter("C6H6", {"C", "H"}) is True
    assert element_filter("CCl4", {"C", "H"}) is False
    assert element_filter("not_a_formula", {"C", "H"}) is False


def test_filter_spec_mol_filters_by_nce_and_ce_type():
    spec_df, mol_df = make_spec_and_mol_frames()

    filtered_spec, filtered_mol = filter_spec_mol(
        spec_df,
        mol_df,
        frag_modes=["hcd"],
        inst_types=["orbitrap"],
        ces="nce",
        ce_types=["stepped"],
        max_prec_mz=200.0,
    )

    assert len(filtered_spec) == 1
    assert filtered_spec.loc[0, "mol_id"] == 1
    assert len(filtered_mol) == 1
    assert filtered_mol.loc[0, "mol_id"] == 1


def test_filter_spec_mol_raises_on_unknown_ce_type():
    spec_df, mol_df = make_spec_and_mol_frames()

    with pytest.raises(ValueError):
        filter_spec_mol(spec_df, mol_df, ce_types=["invalid"])


def test_filter_spec_mol_raises_when_no_entries_remain():
    spec_df, mol_df = make_spec_and_mol_frames()

    with pytest.raises(ValueError):
        filter_spec_mol(spec_df, mol_df, dsets=["missing_dataset"])


def test_filter_spec_mol_sampling_is_deterministic():
    spec_df, mol_df = make_spec_and_mol_frames()
    sampled_first, _ = filter_spec_mol(spec_df, mol_df, num_entries=1)

    spec_df, mol_df = make_spec_and_mol_frames()
    sampled_second, _ = filter_spec_mol(spec_df, mol_df, num_entries=1)

    assert sampled_first.iloc[0]["spec_id"] == sampled_second.iloc[0]["spec_id"]


def test_merge_spec_df_merges_peaks_and_handles_ces():
    spec_df = pd.DataFrame(
        [
            {
                "group_id": 1,
                "peaks": [(10.0, 1.0), (20.0, 2.0)],
                "spec_id": 501,
                "dset": "nist",
                "inst_type": "orbitrap",
                "frag_mode": "hcd",
                "ce_type": "stepped",
                "nce": 10.0,
                "ace": np.nan,
                "ion_mode": "positive",
                "prec_type": "[M+H]+",
                "spec_type": "centroid",
                "prec_mz": 150.0,
                "mol_id": 1,
            },
            {
                "group_id": 1,
                "peaks": [(10.0, 1.0), (30.0, 1.0)],
                "spec_id": 502,
                "dset": "nist",
                "inst_type": "orbitrap",
                "frag_mode": "hcd",
                "ce_type": "stepped",
                "nce": 20.0,
                "ace": 5.0,
                "ion_mode": "positive",
                "prec_type": "[M+H]+",
                "spec_type": "centroid",
                "prec_mz": 150.0,
                "mol_id": 1,
            },
        ]
    )

    merged_with_ces = merge_spec_df(spec_df, renormalize=True, sum_ints=True, keep_ces=True)
    peaks = merged_with_ces.loc[0, "peaks"]

    assert len(merged_with_ces) == 1
    assert peaks == [(10.0, 0.4), (20.0, 0.4), (30.0, 0.2)]

    nce_list = merged_with_ces.loc[0, "nce"]
    ace_list = merged_with_ces.loc[0, "ace"]
    assert nce_list == [10.0, 20.0]
    assert len(ace_list) == 2
    assert np.isnan(ace_list[0])
    assert ace_list[1] == 5.0

    merged_no_ces = merge_spec_df(spec_df, renormalize=False, sum_ints=False, keep_ces=False)
    assert len(merged_no_ces) == 1
    assert merged_no_ces.loc[0, "peaks"] == [(10.0, 1.0), (20.0, 2.0), (30.0, 1.0)]
    assert np.isnan(merged_no_ces.loc[0, "nce"])
    assert np.isnan(merged_no_ces.loc[0, "ace"])
