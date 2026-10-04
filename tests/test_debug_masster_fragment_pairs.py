"""Scientific edge cases for ambiguity-preserving fragment comparisons."""

import numpy as np
import pytest
from rdkit import Chem

from scripts.debug.masster_memorization.debug_masster_base_peak_tables import MoleculeChemistry
from scripts.debug.masster_memorization.debug_masster_fragment_pairs import (
    compare_peaks,
    describe_fragment,
)


def compare(mzs, probs, target_mzs=(100.0, 150.0), target_ints=(0.8, 0.2)):
    return compare_peaks(
        np.array(mzs),
        np.array(probs),
        np.array(target_mzs),
        np.array(target_ints),
        1e-5,
        200.0,
    )


def test_all_compatible_channels_survive_even_zero_probability():
    row, target, winner = compare([99.999, 100.001, 100.0015, 150.0], [0.1, 0.05, 0, 0.85])
    assert row["outcome"] == "under_ranked"
    np.testing.assert_array_equal(target, [True, True, True, False])
    np.testing.assert_array_equal(winner, [False, False, False, True])
    assert row["target_predicted_group_share"] == pytest.approx(0.15)
    assert row["winner_measured_max_share"] == pytest.approx(0.2)
    assert row["strict_measured_preference"]


def test_winner_is_merged_peak_not_largest_channel():
    row, target, winner = compare([100.0, 100.00001, 150.0], [0.3, 0.3, 0.4])
    assert row["outcome"] == "hit"
    assert row["winner_predicted_share"] == pytest.approx(0.6)
    np.testing.assert_array_equal(winner, [True, True, False])
    np.testing.assert_array_equal(target, winner)
    assert row["groups_overlap"]


def test_uncovered_is_not_a_ranking_pair():
    row, target, _ = compare([120.0, 150.0], [0.1, 0.9])
    assert row["outcome"] == "uncovered"
    assert not target.any()
    assert row["target_predicted_group_share"] == 0


def test_measured_ties_and_absent_winner_are_explicit():
    tied, _, _ = compare([100.0, 150.0], [0.1, 0.9], target_ints=(0.5, 0.5))
    assert tied["target_tied_base_count"] == 2
    assert not tied["strict_measured_preference"]
    absent, _, _ = compare([100.0, 180.0], [0.1, 0.9])
    assert absent["winner_measured_peak_count"] == 0
    assert absent["winner_measured_max_share"] == 0


def test_tolerance_uses_measured_mass_and_200_da_floor():
    low, _, _ = compare([100.0019], [1.0])
    high, mask, _ = compare([500.0049, 500.0051], [0.6, 0.4], (500.0,), (1.0,))
    assert low["outcome"] == "hit"
    assert high["target_channel_count"] == 1
    np.testing.assert_array_equal(mask, [True, False])


def test_boundary_bonds_are_original_molecule_indices():
    chem = MoleculeChemistry(Chem.MolFromSmiles("CCO"))
    note = describe_fragment(chem, (0, 1))
    assert note["boundary_bond_indices"] == [1]
    assert note["boundary_atom_pairs"] == [[1, 2]]
    assert note["cut_classes"] == ["alkyl_O"]
    assert note["atom_indices"] == [0, 1]
    assert describe_fragment(chem, (0, 1, 2))["cut_count"] == 0


def test_invalid_data_is_not_silently_skipped():
    with pytest.raises(ValueError, match="nonempty"):
        compare([], [])
    with pytest.raises(ValueError, match="positive sum"):
        compare([100.0], [0.0])
    with pytest.raises(ValueError, match="nonfinite"):
        compare([100.0], [np.nan])


def test_audit_detects_a_missing_zero_share_alternative(tmp_path):
    import json

    import pandas as pd

    from scripts.debug.masster_memorization.validate_masster_fragment_pairs import audit

    mz = np.array([100.0, 100.001, 150.0])
    intensity = np.array([0.1, 0.0, 0.9])
    measured_mz = np.array([100.0, 150.0])
    measured_intensity = np.array([0.8, 0.2])
    row, _, _ = compare(mz, intensity)
    row.update(
        {
            "spec_id": 1,
            "mol_id": 2,
            "tolerance_rel": 1e-5,
            "tolerance_min_mz": 200.0,
            "target_structure_count": 2,
            "winner_structure_count": 1,
        }
    )
    pd.DataFrame([row]).to_csv(tmp_path / "pairs.csv", index=False)
    reference = tmp_path / "reference.pkl"
    pd.to_pickle(
        {
            "spectra": pd.DataFrame(
                [
                    {
                        "spec_id": 1,
                        "mol_id": 2,
                        "pred_mz": mz,
                        "pred_int": intensity,
                        "true_mz": measured_mz,
                        "true_int": measured_intensity,
                    }
                ]
            )
        },
        reference,
    )
    channels = [
        {
            "spec_id": 1,
            "mol_id": 2,
            "channel_index": i,
            "mz": float(mz[i]),
            "predicted_share": float(intensity[i]),
            "target_compatible": i < 2,
            "winner_contributor": i == 2,
            "atom_indices": [i],
        }
        for i in range(3)
    ]
    path = tmp_path / "channels.jsonl"
    path.write_text("".join(json.dumps(ch) + "\n" for ch in channels))
    assert audit(tmp_path, reference)["status"] == "passed"
    # Deleting a zero-share alternative leaves every intensity sum unchanged.
    path.write_text("".join(json.dumps(ch) + "\n" for ch in channels if ch["channel_index"] != 1))
    with pytest.raises(ValueError, match="incomplete group"):
        audit(tmp_path, reference)


def test_rival_is_a_real_alternative_for_hits_and_misses():
    from scripts.debug.masster_memorization.debug_masster_fragment_pairs import strongest_rival

    mz = np.array([100.0, 100.001, 150.0, 180.0])
    target_mz = np.array([100.0, 150.0])
    target_int = np.array([0.8, 0.2])
    for pred in (np.array([0.6, 0.2, 0.15, 0.05]), np.array([0.1, 0.1, 0.7, 0.1])):
        record, mask = strongest_rival(mz, pred, target_mz, target_int, 1e-5, 200.0)
        assert record["rival_mz"] == 150.0
        assert record["rival_measured_max_share"] == pytest.approx(0.2)
        np.testing.assert_array_equal(mask, [False, False, True, False])
    record, mask = strongest_rival(mz[:2], np.array([0.6, 0.4]), target_mz, target_int, 1e-5, 200.0)
    assert not record["rival_available"]
    assert not mask.any()


def test_rival_excludes_whole_rounded_group_when_any_channel_matches_base():
    from scripts.debug.masster_memorization.debug_masster_fragment_pairs import strongest_rival

    record, mask = strongest_rival(
        np.array([100.00199, 100.00201, 150.0]),
        np.array([0.4, 0.4, 0.2]),
        np.array([100.0]),
        np.array([1.0]),
        1e-5,
        200.0,
    )
    assert record["rival_mz"] == 150.0
    np.testing.assert_array_equal(mask, [False, False, True])
