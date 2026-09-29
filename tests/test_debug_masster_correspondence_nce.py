"""Energy selection must preserve molecular triples and report missing common energies."""

import numpy as np
import pandas as pd
import pytest

from scripts.debug.debug_masster_correspondence_nce import select_banded_pairs, summarize_bands


def fixture_data():
    spectra = pd.DataFrame(
        [
            {
                "spec_id": m * 100 + n,
                "mol_id": m,
                "nce": float(n),
                "inst_type": "OB",
                "prec_type": "[M+H]+",
                "frag_mode": "HCD",
                "ce_type": "single",
            }
            for m, energies in [(1, [10, 20, 30, 50]), (2, [10, 20, 30, 50]), (3, [10, 20, 30])]
            for n in energies
        ]
    )
    anchor = pd.DataFrame(
        [
            {
                "query_spec": 110,
                "donor_spec": sid,
                "kind": kind,
                "novel": True,
                "query_max_tanimoto": 0.3,
            }
            for sid, kind in [(210, "neighbor"), (310, "control")]
        ]
    )
    return spectra, anchor


def test_band_selection_uses_common_observed_energy_and_fixed_triples():
    spectra, anchor = fixture_data()
    pairs, availability = select_banded_pairs(spectra, anchor, [[0, 20], [20, 40], [40, 60]])
    assert pairs.nce.tolist() == [10.0, 10.0, 30.0, 30.0]
    assert pairs.donor_spec.tolist() == [210, 310, 230, 330]
    assert availability.available.tolist() == [True, True, False]
    assert np.isnan(availability.nce.iloc[-1])
    spectra["peaks"] = ["arbitrary ignored labels"] * len(spectra)
    pd.testing.assert_frame_equal(
        pairs, select_banded_pairs(spectra, anchor, [[0, 20], [20, 40], [40, 60]])[0]
    )


def test_tie_break_replicates_and_acquisition_exclusion():
    spectra, anchor = fixture_data()
    spectra = pd.concat([spectra, spectra[spectra.spec_id == 220].assign(spec_id=999)])
    pairs, _ = select_banded_pairs(spectra, anchor, [[10, 40]])
    assert pairs.nce.eq(20).all()  # 20 and 30 are equidistant from 25.
    assert 999 not in pairs.donor_spec.values
    spectra.loc[spectra.spec_id == 310, "inst_type"] = "OB_F"
    with pytest.raises(ValueError, match="acquisition"):
        select_banded_pairs(spectra, anchor, [[0, 20]])


def test_common_cohort_excludes_molecules_with_missing_bands():
    spectra, anchor = fixture_data()
    pairs, availability = select_banded_pairs(spectra, anchor, [[0, 20], [20, 40]])
    stats = pairs[["query_spec", "donor_spec", "kind", "query_mol", "novel"]].copy()
    stats["view"] = "ion"
    stats["two_reliable"] = [True, False, True, False]
    stats["matches"] = 4
    stats["query_covered_intensity"] = 0.5
    result = summarize_bands(stats, pairs, availability)
    common = result[(result.cohort == "all_bands_common") & (result.instrument == "all")]
    assert len(common) == 4
    assert common.molecules.eq(1).all()
    assert common.loc[common.kind == "neighbor", "fraction"].eq(1).all()
    extra = availability.iloc[:1].copy()
    extra["available"] = False
    availability = pd.concat([availability, extra])
    assert "all_bands_common" not in summarize_bands(stats, pairs, availability).cohort.values
