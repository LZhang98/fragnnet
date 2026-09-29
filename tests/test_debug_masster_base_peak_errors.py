"""Tests for the base-peak error probe and its chemistry tables."""

from collections import Counter

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

import scripts.debug.debug_masster_base_peak_errors as PROBE
import scripts.debug.debug_masster_base_peak_tables as TABLES


def test_merge_channels_sums_same_mz() -> None:
    mzs = np.array([150.1, 100.05, 100.05 + 1e-9, 120.0])
    probs = np.array([0.1, 0.2, 0.3, 0.4])
    unique, merged, inverse = PROBE.merge_channels(mzs, probs)
    np.testing.assert_allclose(unique, [100.05, 120.0, 150.1])
    np.testing.assert_allclose(merged, [0.5, 0.4, 0.1])
    np.testing.assert_array_equal(inverse, [2, 0, 0, 1])


def test_map_targets_uses_floor_and_nearest() -> None:
    cands = np.array([100.000, 100.003, 300.000])
    # 10 ppm with a 200 Da floor is 0.002 Da below m/z 200 and 0.003 Da at 300.
    peaks = np.array([100.0012, 100.0035, 300.0029, 150.0])
    ints = np.array([1.0, 2.0, 3.0, 4.0])
    on_cand, peak_to_cand = PROBE.map_targets(peaks, ints, cands, 1e-5, 200.0)
    np.testing.assert_array_equal(peak_to_cand, [0, 1, 2, -1])
    np.testing.assert_allclose(on_cand, [1.0, 2.0, 3.0])


def test_cosine_decomposition_parts() -> None:
    target = np.array([0.6, 0.3, 0.1])
    norm = float(np.linalg.norm(np.append(target, 0.2)))  # one uncovered peak of 0.2
    # Order-preserving but flattened: no ranking error, only calibration.
    flat = PROBE.cosine_decomposition(np.array([0.4, 0.35, 0.25]), target, norm)
    assert flat["coverage"] == pytest.approx(1 - np.linalg.norm(target) / norm)
    assert flat["ranking"] == pytest.approx(0.0, abs=1e-12)
    assert flat["calibration"] > 0
    # Reversed order: ranking error appears; the three parts always sum to 1 - cos.
    reverse = PROBE.cosine_decomposition(np.array([0.1, 0.3, 0.6]), target, norm)
    assert reverse["ranking"] > 0.1
    total = reverse["coverage"] + reverse["ranking"] + reverse["calibration"]
    assert total == pytest.approx(1 - reverse["cos"])


def test_base_peak_outcome_labels() -> None:
    cands = np.array([100.0, 200.0])
    peaks, ints = np.array([100.0, 200.0]), np.array([1.0, 0.5])
    mapped = np.array([0, 1])
    assert (
        PROBE.base_peak_outcome(peaks, ints, mapped, cands, np.array([0.7, 0.3]), 1e-5, 200)
        == "hit"
    )
    assert (
        PROBE.base_peak_outcome(peaks, ints, mapped, cands, np.array([0.3, 0.7]), 1e-5, 200)
        == "under_ranked"
    )
    assert (
        PROBE.base_peak_outcome(
            peaks, ints, np.array([-1, 1]), cands, np.array([0.3, 0.7]), 1e-5, 200
        )
        == "uncovered"
    )


def test_descending_ranks_is_stable() -> None:
    np.testing.assert_array_equal(PROBE.descending_ranks(np.array([0.2, 0.5, 0.2])), [2, 1, 3])


@pytest.mark.parametrize(
    ("counts", "formula"),
    [(Counter(H=2, O=1), "H2O"), (Counter(H=3, N=1), "H3N"), (Counter(C=2, H=2, O=1), "C2H2O")],
)
def test_hill_round_trip(counts: Counter, formula: str) -> None:
    assert TABLES.hill(counts) == formula
    assert TABLES.parse_formula(formula) == counts


def test_formula_difference() -> None:
    assert TABLES.formula_difference("C2H4O2", "C2H2O") == "H2O"
    assert TABLES.formula_difference("H2O", "H2O") == "same"
    assert TABLES.formula_difference("H2O", "H3N") == "mixed"


@pytest.fixture
def acetanilide() -> "TABLES.MoleculeChemistry":
    """CC(=O)Nc1ccccc1: atoms 0-2 acetyl (C, C, O), 3 N, 4-9 the ring."""
    return TABLES.MoleculeChemistry(Chem.MolFromSmiles("CC(=O)Nc1ccccc1"))


def test_amide_bond_class_and_ketene_loss(acetanilide: "TABLES.MoleculeChemistry") -> None:
    amide = acetanilide.bonds.index((1, 3))
    assert acetanilide.bond_class[amide] == "amide_CN"
    # Anilinium: N plus the ring, one H moved onto the fragment; the loss is ketene.
    note = acetanilide.annotate((3, 4, 5, 6, 7, 8, 9), 1)
    assert note["loss_formula"] == "C2H2O"
    assert note["loss_class"] == "ketene"
    assert note["cut_classes"] == ("amide_CN",)
    assert note["keeps_aromatic"] and not note["ring_cut"] and not note["is_precursor"]
    assert acetanilide.annotate(tuple(range(10)), 0)["is_precursor"]


def test_ring_cut_is_flagged(acetanilide: "TABLES.MoleculeChemistry") -> None:
    note = acetanilide.annotate((0, 1, 2, 3, 4, 5), 0)
    assert note["ring_cut"]
    assert all(c.startswith("ring:") for c in note["cut_classes"])
    assert not note["keeps_aromatic"]


def test_peak_attributes_flags_ambiguous_cut(acetanilide: "TABLES.MoleculeChemistry") -> None:
    # Two structures of the same formula cut different bonds: the peak is ambiguous.
    structures = [
        ((1, 2, 3, 4, 5, 6, 7, 8, 9), 0, 0, 0.6),
        ((0, 1, 2, 3, 4, 5, 6, 7, 8), 0, 0, 0.4),
    ]
    out = TABLES.peak_attributes(structures, acetanilide, {})
    assert out["ambiguous_cut"]
    assert out["n_structures"] == 2
    assert out["keeps_aromatic"] == pytest.approx(0.6)


def test_pairwise_bias_sign() -> None:
    # The model gets every pair right when X is on #1 and wrong when X is on #2: it favors X.
    n = 40
    frame = pd.DataFrame({"correct": [True] * n + [False] * n, "mol_id": np.arange(2 * n) % 10})
    first = np.array([True] * n + [False] * n)
    stats = TABLES.pairwise_bias(frame, first, ~first, reps=50)
    assert stats["bias"] == pytest.approx(1.0)
    assert stats["n_on_first"] == n and stats["n_on_second"] == n
    assert stats["lo"] == pytest.approx(1.0) and stats["hi"] == pytest.approx(1.0)


@pytest.fixture
def phenethylamine() -> "TABLES.MoleculeChemistry":
    """NCCc1ccccc1: atom 0 the aliphatic amine, 1-2 the chain, 3-8 the ring."""
    return TABLES.MoleculeChemistry(Chem.MolFromSmiles("NCCc1ccccc1"))


def test_basic_site_needs_a_real_base(
    acetanilide: "TABLES.MoleculeChemistry", phenethylamine: "TABLES.MoleculeChemistry"
) -> None:
    # The amide N is not a basic site, and carbonyls no longer stand in for one.
    assert acetanilide.base_class == "none"
    assert not acetanilide.annotate((3, 4, 5, 6, 7, 8, 9), 1)["keeps_base"]
    assert phenethylamine.base_class == "aliphatic_amine"
    assert phenethylamine.annotate((0, 1, 2), 0)["keeps_base"]
    tropylium = phenethylamine.annotate((2, 3, 4, 5, 6, 7, 8), 0)
    assert not tropylium["keeps_base"] and not tropylium["loses_aromatic"]
    assert phenethylamine.annotate((0, 1, 2), 0)["loses_aromatic"]


@pytest.mark.parametrize(
    "atoms",
    [(0, 1, 2), (2, 3, 4, 5, 6, 7, 8), (0, 1, 2, 3, 4), tuple(range(9)), (1, 2, 3, 4, 5, 6, 7, 8)],
)
def test_vectorized_attributes_match_annotate(
    phenethylamine: "TABLES.MoleculeChemistry", atoms: tuple
) -> None:
    inside = np.zeros((1, phenethylamine.n_atoms), dtype=bool)
    inside[0, list(atoms)] = True
    fragment = PROBE.fragment_attributes(inside, phenethylamine)
    for h_shift in (-1, 0):  # +H is impossible on the intact molecule
        x = PROBE.channel_attributes(fragment, np.array([0]), np.array([h_shift]), 9)[0]
        note = phenethylamine.annotate(atoms, h_shift)
        expected = {**note, "h_shift_abs": abs(h_shift)}
        for value, name in zip(x, PROBE.CANDIDATE_ATTRIBUTES, strict=True):
            assert value == pytest.approx(float(expected[name])), name


@pytest.fixture
def candidate_arrays() -> dict:
    """Two spectra of 3 and 2 candidates, one attribute column, uncovered target in spectrum 0."""
    rng = np.random.default_rng(0)
    pred = np.array([0.5, 0.3, 0.2, 0.9, 0.1])
    tgt = np.array([0.2, 0.6, 0.1, 0.4, 0.5])
    return {
        "log_pred": np.log(pred),
        "tgt": tgt,
        "x": rng.normal(size=(5, 1)),
        "offsets": np.array([0, 3, 5]),
        "target_norm": np.array([np.linalg.norm([0.2, 0.6, 0.1, 0.3]), np.linalg.norm([0.4, 0.5])]),
        "pred": pred,
    }


def test_reweighted_cosine_identity_and_gradient(candidate_arrays: dict) -> None:
    a = candidate_arrays
    args = (a["log_pred"], a["tgt"], a["x"], a["offsets"], a["target_norm"])
    cos, _ = TABLES.reweighted_cosine(np.array([1.0, 0.0]), *args)
    expected = [
        a["pred"][:3] @ a["tgt"][:3] / (np.linalg.norm(a["pred"][:3]) * a["target_norm"][0]),
        a["pred"][3:] @ a["tgt"][3:] / (np.linalg.norm(a["pred"][3:]) * a["target_norm"][1]),
    ]
    np.testing.assert_allclose(cos, expected)
    theta = np.array([0.7, 0.3])
    _, grad = TABLES.reweighted_cosine(theta, *args)
    step = 1e-6
    numeric = [
        (
            TABLES.reweighted_cosine(theta + step * np.eye(2)[i], *args)[0].mean()
            - TABLES.reweighted_cosine(theta - step * np.eye(2)[i], *args)[0].mean()
        )
        / (2 * step)
        for i in range(2)
    ]
    np.testing.assert_allclose(grad, numeric, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        ((0, 1, 2), (0, 1, 2), "same_atoms"),
        ((0, 1, 2), (3, 4, 5, 6, 7, 8), "complementary"),
        ((0, 1), (3, 4, 5, 6, 7, 8), "disjoint"),
        ((3, 4, 5, 6, 7, 8), (2, 3, 4, 5, 6, 7, 8), "first_inside_second"),
        ((2, 3, 4, 5, 6, 7, 8), (3, 4, 5, 6, 7, 8), "second_inside_first"),
        ((0, 1, 2, 3), (2, 3, 4), "overlapping"),
    ],
)
def test_pair_relation(first: tuple, second: tuple, expected: str) -> None:
    assert TABLES.pair_relation(first, second, 9) == expected
