"""Checks for the first-stage intensity reliability census."""

import pandas as pd
import pytest

from scripts.eval.masster_intensity_reliability import audit


def test_census_flags_identical_copies_without_claiming_independence():
    rows = []
    for spec_id, mol_id, peaks in (
        (1, 1, [(100.0, 2.0)]),
        (2, 1, [(100.0, 2.0)]),
        (3, 2, [(110.0, 3.0)]),
    ):
        rows.append(
            {"spec_id": spec_id, "mol_id": mol_id, "peaks": peaks,
                 "prec_type": "[M+H]+", "inst_type": "ORB",
                 "frag_mode": "HCD", "dset": "nist20_hr",
                 "dset_spec_id": str(spec_id), "col_gas": "N2",
                 "res": 2, "ce_type": "single", "ace": 20.0,
                 "nce": float("nan")}
        )
    spectra = pd.DataFrame(rows)
    splits = {
        "train": spectra.loc[:1, ["spec_id", "mol_id"]],
        "val": spectra.loc[2:, ["spec_id", "mol_id"]],
    }
    membership, summary = audit(spectra, splits)
    assert summary["replicate_groups"] == 1
    assert summary["strict_independent_groups"] == 0
    assert membership["identical_copy"].tolist() == [True, True, False]


def test_census_rejects_primary_molecule_leakage():
    spectra = pd.DataFrame(
        [{"spec_id": i, "mol_id": 1, "peaks": [], "prec_type": "[M+H]+",
             "inst_type": "ORB", "frag_mode": "HCD", "dset": "nist20_hr",
             "dset_spec_id": str(i), "col_gas": "N2", "res": 2,
             "ce_type": "single", "ace": 20.0, "nce": float("nan")} for i in (1, 2)]
    )
    splits = {
        "train": pd.DataFrame({"spec_id": [1], "mol_id": [1]}),
        "val": pd.DataFrame({"spec_id": [2], "mol_id": [1]}),
    }
    with pytest.raises(ValueError, match="overlap"):
        audit(spectra, splits)
