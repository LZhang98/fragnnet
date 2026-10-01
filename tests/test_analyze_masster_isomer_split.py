"""Focused checks for the checkpoint isomer-split readout."""

import pandas as pd
import pytest

from scripts.debug.analyze_masster_isomer_split import (
    collapse_structures,
    compare_splits,
    cut_preference_read,
)


def test_collapse_structures_sums_duplicate_channels() -> None:
    structures = [((0, 1), 0, 0, 0.2), ((0, 1), 0, 0, 0.3), ((1, 2), 0, 0, 0.5)]
    assert collapse_structures(structures) == {((0, 1), 0, 0): 0.5, ((1, 2), 0, 0): 0.5}


def test_compare_splits_requires_shared_structures_and_reports_change() -> None:
    first = ((0, 1), 0, 0)
    second = ((1, 2), 0, 0)
    left = pd.DataFrame([{
        "spec_id": 4, "mz": 100.0, "pred": 0.5, "tgt": 0.4,
        "structures": [(*first, 0.4), (*second, 0.1)],
    }])
    right = pd.DataFrame([{
        "spec_id": 4, "mz": 100.0, "pred": 0.5, "tgt": 0.4,
        "structures": [(*first, 0.1), (*second, 0.4)],
    }])
    summary, matched = compare_splits(left, right)
    assert summary["totals_within_10pct"]["peaks"] == 1
    assert summary["totals_within_10pct"]["top_structure_agreement"] == 0
    assert matched.l1_half.iloc[0] == pytest.approx(0.6)


def test_cut_preference_crossfits_other_molecules() -> None:
    rows = []
    for mol_id in range(20):
        spec_id = mol_id + 100
        for cut, target in ((0, 0.2), (1, 0.8)):
            rows.append({
                "spec_id": spec_id, "mol_id": mol_id, "mz": 50 + cut,
                "target": target, "prob": target, "n_structures": 1,
                "n_cuts": 1, "frac_atoms": 0.5, "h_shift_abs": 0,
                "is_precursor": 0, "cut_A": cut,
            })
        for cut, prob in ((0, 0.3), (1, 0.7)):
            rows.append({
                "spec_id": spec_id, "mol_id": mol_id, "mz": 80,
                "target": 0.5, "prob": prob, "n_structures": 2,
                "n_cuts": 1, "frac_atoms": 0.5, "h_shift_abs": 0,
                "is_precursor": 0, "cut_A": cut,
            })
    result = cut_preference_read(pd.DataFrame(rows), ["cut_A"])
    assert result["ambiguous_test_peaks"] == 20
    assert result["top_hit_with_cut_classes"] == pytest.approx(1.0)
    assert result["cut_increment"] == pytest.approx(1.0)
