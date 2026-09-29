"""Structural matching must preserve ambiguity and avoid intensity-selected explanations."""

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

from scripts.debug.debug_masster_structural_alignment import (
    annotate_groups,
    pair_audit,
    strict_edges,
    structural_signatures,
    summarize,
)


def make_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "group": [0, 1, 2],
            "eligible": [True, True, True],
            "ion": ["C1H3", "C2H5", "C3H7"],
            "target": [0.6, 0.3, 0.1],
            "pred": [0.3, 0.6, 0.1],
            "structure_keys": [("a",), ("b",), ("c",)],
            "context_keys": [("x",), ("y",), ("z",)],
        }
    )


def test_same_formula_different_connectivity_and_cut_environment():
    ethanol, ether = Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("COC")
    assert (
        structural_signatures(ethanol, (0, 1, 2), 1)[0]
        != structural_signatures(ether, (0, 1, 2), 1)[0]
    )
    amine = Chem.MolFromSmiles("CCN")
    a, b = structural_signatures(ethanol, (0, 1), 1), structural_signatures(amine, (0, 1), 1)
    assert a[0] == b[0]
    assert a[1] != b[1]  # Same retained ethyl connectivity, different discarded-side chemistry.


def test_signatures_are_atom_order_invariant_including_open_aromatic_fragments():
    mol = Chem.MolFromSmiles("CCOc1ccccc1")
    permutation = [5, 2, 0, 8, 3, 1, 7, 4, 6]
    other = Chem.RenumberAtoms(mol, permutation)
    for kept in [(0, 1), (3, 4, 5), (2, 3, 4, 5, 6, 7, 8)]:
        mapped = tuple(permutation.index(k) for k in kept)
        assert structural_signatures(mol, kept, 1) == structural_signatures(other, mapped, 1)


def test_strict_matching_excludes_ambiguity_without_selecting_best_explanation():
    left, right = make_frame(), make_frame()
    left.at[0, "structure_keys"] = ("a", "alternative")
    right.at[1, "context_keys"] = ("different",)
    assert strict_edges(left, right, "formula") == [(0, 0), (1, 1), (2, 2)]
    assert strict_edges(left, right, "connectivity") == [(1, 1), (2, 2)]
    assert strict_edges(left, right, "cut_context") == [(2, 2)]
    left["target"], left["pred"] = [0, 0, 1], [1, 0, 0]
    assert strict_edges(left, right, "cut_context") == [(2, 2)]


def test_equivalent_atom_sets_collapse_but_channel_alternatives_remain():
    mol = Chem.MolFromSmiles("CCC")
    data = make_frame().iloc[:1].copy()
    data.attrs.update(
        atom_numbers=np.array([6, 6, 6]),
        atom_hydrogens=np.array([3, 2, 3]),
        chemical_bonds=np.array([[0, 1], [1, 2]]),
        path_atoms=[(0,), (2,)],
        path_modes=[0, 0],
        path_h_shifts=[0, 0],
        path_group=[0, 0],
    )
    result = annotate_groups(data, mol, 1, {}, {})
    assert len(result.iloc[0].structure_keys) == len(result.iloc[0].context_keys) == 1
    data.attrs["path_modes"] = [0, 1]
    result = annotate_groups(data, mol, 1, {}, {})
    assert len(result.iloc[0].structure_keys) == 1
    assert len(result.iloc[0].context_keys) == 2
    data.attrs["atom_numbers"] = np.array([6, 7, 6])
    with pytest.raises(ValueError, match="atom order"):
        annotate_groups(data, mol, 1, {}, {})


def test_empty_matches_do_not_become_perfect_agreement_and_controls_preserve_size():
    cfg = {"control_repeats": 20, "reliable_intensity_fraction": 0.01}
    left, right = make_frame(), make_frame()
    records = pair_audit(left, right, cfg, 42)
    for record in records:
        assert record["matches"] == 3
        assert record["measured_cosine"] == pytest.approx(1)
        assert record["residual_cosine"] == pytest.approx(1)
        assert record["measured_minus_count"] == pytest.approx(0)
        assert record["residual_minus_shuffle"] > 0
    right["structure_keys"] = [("other",)] * 3
    records = pair_audit(left, right, cfg, 42)
    for record in records[1:]:
        assert record["matches"] == 0 and record["query_intensity"] == 0
        assert np.isnan(record["measured_cosine"])
        assert np.isnan(record["residual_minus_count"])


def test_bootstrap_weights_query_identity_not_number_of_conditions():
    cfg = {
        "control_repeats": 2,
        "reliable_intensity_fraction": 0.01,
        "seed": 42,
        "bootstrap_repeats": 20,
    }
    record = pair_audit(make_frame(), make_frame(), cfg, 42)[0]
    records = pd.DataFrame(
        [
            {
                **record,
                "novel": True,
                "kind": "neighbor",
                "query_mol": mid,
                "query_intensity": intensity,
            }
            for mid, intensity in [(1, 0.0), (1, 0.0), (2, 1.0)]
        ]
    )
    summary = summarize(records, cfg)
    row = summary[summary.metric == "query_intensity"].iloc[0]
    assert row["mean"] == pytest.approx(0.5)
    assert row.query_identities == 2 and row.pairs == 3


def test_shared_absence_cannot_pass_measured_label_sanity_controls():
    left, right = make_frame(), make_frame()
    left["target"] = 0.0
    right["target"] = 0.0
    cfg = {"control_repeats": 20, "reliable_intensity_fraction": 0.01}
    for record in pair_audit(left, right, cfg, 42):
        assert np.isnan(record["measured_cosine"])
        assert record["residual_cosine"] == pytest.approx(1)
        assert record["residual_minus_shuffle"] > 0  # Original metric has this confound.
        assert record["residual_minus_zero_reference"] == pytest.approx(0)
        assert record["residual_minus_label_only_shuffle"] == pytest.approx(0)
