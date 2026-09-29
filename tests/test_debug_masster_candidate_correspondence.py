"""Fixed-rule matching must not invent chemical assignments or reuse peak intensity."""

import numpy as np
import pandas as pd
import pytest
import torch as th

from scripts.debug.debug_masster_candidate_correspondence import (
    candidate_groups,
    match_groups,
    reconstruct_formulas,
    select_pairs,
    summarize_pairs,
)


def formula(**counts):
    values = np.zeros(119, dtype=int)
    for key, number in counts.items():
        values[{"H": 1, "C": 6, "N": 7, "O": 8}[key]] = number
    return values


def groups(masses=(100.0, 120.0), forms=None, truth=(100.0, 120.0), intensities=(3.0, 1.0)):
    if forms is None:
        forms = np.array([formula(C=6, H=12, O=1), formula(C=7, H=14, O=1)])
    return candidate_groups(
        np.array(masses),
        np.ones(len(masses)),
        np.array(forms),
        formula(C=10, H=20, O=3),
        [(i,) for i in range(len(masses))],
        np.array(truth),
        np.array(intensities),
        200.0,
        1e-5,
        200.0,
        1.5,
    )


def test_water_loss_matches_different_fragment_formulas():
    a = groups((100.0,), [formula(C=10, H=18, O=2)], (100.0,), (1.0,))
    b = candidate_groups(
        np.array([114.0]),
        np.ones(1),
        np.array([formula(C=11, H=20, O=2)]),
        formula(C=11, H=22, O=3),
        [(0,)],
        np.array([114.0]),
        np.ones(1),
        250.0,
        1e-5,
        200.0,
        1.5,
    )
    assert a.iloc[0].loss == "H2O1"
    assert match_groups(a, b, "ion") == []
    assert match_groups(a, b, "loss") == [(0, 0)]


def test_ambiguous_formula_and_overlapping_mass_windows_are_excluded():
    a = groups((100.0, 100.0), [formula(C=6), formula(C=5, N=1)])
    assert a.iloc[0].formula_alternatives == 2
    assert not a.iloc[0].eligible
    b = groups((100.0, 100.001), truth=(100.0005,), intensities=(1.0,))
    assert b.unresolved.all()
    assert not b.eligible.any()
    assert b.target.sum() == 0  # A peak compatible with both groups is not assigned twice.


def test_structural_alternatives_keep_one_observable_group():
    a = groups((100.0, 100.0), [formula(C=6), formula(C=6)], (100.0,), (1.0,))
    assert len(a) == 1
    assert a.iloc[0].structural_alternatives == 2
    assert a.iloc[0].target == 1.0
    assert a.iloc[0].pred == 1.0
    assert a.iloc[0].eligible


def test_precursor_and_negative_losses_cannot_create_false_loss_matches():
    a = groups((199.0, 120.0), [formula(C=10, H=20, O=3), formula(C=11, H=20, O=3)])
    assert a.loc[a.mz == 199.0, "eligible"].item() is False
    assert a.loc[a.mz == 120.0, "loss"].item() == ""


def test_union_drops_conflicts_and_mapping_is_independent_of_labels():
    a = groups()
    b = groups()
    b["loss"] = list(reversed(b.loss.tolist()))
    assert len(match_groups(a, b, "ion")) == 2
    assert len(match_groups(a, b, "loss")) == 2
    assert match_groups(a, b, "union") == []
    before = match_groups(a, b, "ion")
    a["target"] = [0.0, 1.0]
    b["pred"] = [1.0, 0.0]
    assert match_groups(a, b, "ion") == before


def test_coverage_keeps_zero_labels_and_reliable_pair_threshold():
    a, b = groups(), groups(intensities=(1.0, 0.0))
    pairs = pd.DataFrame(
        [{"query_spec": 1, "donor_spec": 2, "query_mol": 3, "kind": "neighbor", "novel": True}]
    )
    stats, matches = summarize_pairs(pairs, {1: a, 2: b}, 0.01)
    assert (stats.matches == 2).all()
    assert (stats.both_reliable == 1).all()
    assert not stats.two_reliable.any()
    assert (matches.donor_target == 0).any()
    assert stats.query_covered_intensity.tolist() == [1.0, 1.0, 1.0]


def test_selection_reproducible_balanced_and_label_blind():
    census = pd.DataFrame(
        [
            {
                "query_mol": i,
                "query_spec": 10 + i,
                "donor_spec": 20 + i,
                "neighbor_rank": 1,
                "controls": [30 + i],
                "query_max_tanimoto": t,
            }
            for i, t in enumerate([0.2, 0.3, 0.7, 0.8])
        ]
    )
    a = select_pairs(census, 1, 12)
    census["fragment_cos"] = [1.0, 0.0, 0.0, 1.0]
    pd.testing.assert_frame_equal(a, select_pairs(census, 1, 12))
    assert len(a) == 4
    assert a.groupby("novel").query_mol.nunique().tolist() == [1, 1]


def test_formula_reconstruction_matches_real_model_channels():
    from test_masster_pl import _build_pl, _escape_inputs

    model, _ = _build_pl()
    model.eval()
    inputs = _escape_inputs("CC(=O)O")
    inputs["spec_prec_type"][:] = model.model.precursor_types.index("[M+H]+")
    with th.inference_mode():
        output = model.model(**inputs, return_annotation=True)
    batch = dict(inputs, batch_size=2)
    formulas, precursors, error = reconstruct_formulas(batch, output, model.model.precursor_types)
    assert np.array_equal(precursors[0], formula(C=2, H=5, O=2))
    assert (formulas >= 0).all()
    assert error < 2e-5
    output["pred_mzs"] = output["pred_mzs"] + 0.01
    with pytest.raises(ValueError, match="Formula/mass"):
        reconstruct_formulas(batch, output, model.model.precursor_types)


def test_high_nce_followup_preserves_query_identities_without_using_intensities():
    census = pd.DataFrame(
        [
            {
                "query_mol": m,
                "query_spec": m * 10 + e,
                "donor_spec": 100 + m * 10 + e,
                "neighbor_rank": 1,
                "controls": [200 + m * 10 + e],
                "query_max_tanimoto": 0.2 if m < 2 else 0.7,
                "nce": e * 10,
            }
            for m in range(4)
            for e in [1, 2]
        ]
    )
    low = select_pairs(census, 1, 12)
    high = select_pairs(census, 1, 12, "highest_nce")
    assert set(low.query_mol) == set(high.query_mol)
    assert (low.query_spec % 10 == 1).all()
    assert (high.query_spec % 10 == 2).all()
    with pytest.raises(ValueError, match="condition"):
        select_pairs(census, 1, 12, "pick_best_intensity")
