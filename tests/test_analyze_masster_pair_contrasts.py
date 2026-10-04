"""Check ambiguity and matching safeguards for the conditional pair analysis."""

import numpy as np
import pandas as pd
import pytest

from scripts.debug.masster_memorization.analyze_masster_pair_contrasts import (
    atom_relation,
    bh_adjust,
    definite_direction,
    matched_effect,
)


def test_definite_direction_does_not_average_ambiguous_structures():
    assert np.isnan(definite_direction([1, 3], [2]))
    assert definite_direction([3, 4], [1, 2]) == 1
    assert definite_direction([1], [2, 3]) == -1
    assert definite_direction([2, 2], [2]) == 0
    assert np.isnan(definite_direction([], [2]))


def test_relationship_distinguishes_complements_and_other_disjoint_sets():
    assert atom_relation(frozenset([0]), frozenset([0, 1]), 3) == "nested"
    assert atom_relation(frozenset([0]), frozenset([1, 2]), 3) == "complementary"
    assert atom_relation(frozenset([0]), frozenset([1]), 3) == "disjoint"
    assert atom_relation(frozenset([0, 1]), frozenset([1, 2]), 3) == "overlapping"


def test_matched_effect_removes_repeat_spectrum_weighting():
    data = pd.DataFrame(
        {
            "mol_id": [1, 2, 3, 4],
            "stratum_key": ["a"] * 4,
            "feature": [1.0, 1.0, -1.0, -1.0],
            "failure": [1.0, 0.0, 0.0, 0.0],
        }
    )
    result = matched_effect(data, "feature", ["stratum_key"])
    duplicated = pd.concat([data, pd.concat([data.iloc[[0]]] * 30)], ignore_index=True)
    repeat = matched_effect(duplicated, "feature", ["stratum_key"])
    assert result["difference"] == pytest.approx(0.5)
    assert result["molecules"] == 4
    assert repeat["difference"] == pytest.approx(result["difference"])
    assert repeat["se"] == pytest.approx(result["se"])


def test_common_support_requires_different_molecules():
    data = pd.DataFrame(
        {
            "mol_id": [1, 2, 3, 4],
            "stratum_key": ["a", "a", "b", "b"],
            "feature": [1.0, 1.0, -1.0, -1.0],
            "failure": [1.0, 0.0, 0.0, 0.0],
        }
    )
    assert matched_effect(data, "feature", ["stratum_key"])["strata"] == 0
    both = pd.DataFrame(
        {
            "mol_id": [1, 2, 1, 2],
            "stratum_key": ["a"] * 4,
            "feature": [1.0, 1.0, -1.0, -1.0],
            "failure": [1.0, 0.0, 0.0, 0.0],
        }
    )
    assert matched_effect(both, "feature", ["stratum_key"])["strata"] == 0
    other = pd.DataFrame(
        {
            "mol_id": [3, 4, 5, 6],
            "stratum_key": ["a"] * 4,
            "feature": [1.0, 1.0, -1.0, -1.0],
            "failure": [1.0, 0.0, 0.0, 0.0],
        }
    )
    result = matched_effect(pd.concat([both, other]), "feature", ["stratum_key"])
    assert result["molecules"] == 4
    assert result["rows"] == 4
    assert result["difference"] == pytest.approx(0.5)


def test_multiple_testing_preserves_missing_entries_and_order():
    q = bh_adjust(pd.Series([0.04, np.nan, 0.001, 0.03]))
    np.testing.assert_allclose(q.iloc[[0, 2, 3]], [0.04 * 4 / 3, 0.004, 0.04 * 4 / 3])
    assert np.isnan(q.iloc[1])


def test_group_consensus_retains_low_probability_ambiguity():
    from scripts.debug.masster_memorization.analyze_masster_pair_contrasts import summarize_groups

    def channel(atoms, cuts, probability):
        return {
            "atom_indices": atoms,
            "atom_count": len(atoms),
            "atom_fraction": len(atoms) / 4,
            "cut_count": cuts,
            "cut_classes": ["alkyl_CC"] * cuts,
            "h_shift": 0,
            "ring_cut": False,
            "breaks_aromatic": False,
            "loses_aromatic": False,
            "keeps_base": False,
            "predicted_share": probability,
        }

    target = [channel([0], 1, 0.999), channel([0, 1], 3, 0.0)]
    rival = [channel([0, 1, 2], 2, 0.1)]
    summary = summarize_groups(target, rival)
    assert np.isnan(summary["cut_count"])
    assert summary["choice_key"] is None
    assert summary["relation"] == "nested"
    assert summary["fragment_only_pair"]
    assert not summary["single_structure_pair"]
    assert summary["atom_fraction"] == -1
