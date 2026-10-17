import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rdkit")

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "debug" / "masster_memorization" / "bond_local_learner.py"
SPEC = importlib.util.spec_from_file_location("bond_local_learner", SCRIPT_PATH)
BL = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BL)


def test_bond_features_symmetric_and_released_piece():
    # Phenyl acetate: bond 3 is the ester O-aryl bond (atoms 3-4), bond 1 is C=O.
    df, labels = BL.bond_features("CC(=O)Oc1ccccc1")
    assert labels[:4] == ["C-C", "C=O", "C-O", "C-O"] and labels[-1] == BL.PRECURSOR
    # Cutting O-aryl releases the acetoxy piece C2H3O2 (lighter than phenyl C6H5).
    row = df.loc[3]
    assert (row.small_heavy, row.small_C, row.small_O, row.small_H) == (4, 2, 2, 3)
    assert row.in_ring == 0 and np.isnan(df.loc[4, "small_heavy"])  # ring bonds release no piece
    # Same molecule written from the other end gives the same features for the same bond.
    df2, labels2 = BL.bond_features("c1ccc(OC(C)=O)cc1")
    ester = [i for i, lab in enumerate(labels2) if lab == "C-O"]
    rows2 = df2.loc[ester].drop(columns=["n_heavy", "mw"]).round(6)
    rows1 = df.loc[[2, 3]].drop(columns=["n_heavy", "mw"]).round(6)
    key = lambda d: sorted(map(tuple, d[sorted(d.columns)].fillna(-9).to_numpy()))  # noqa: E731
    assert key(rows1) == key(rows2)
    assert df.loc[-1, "is_precursor"] == 1.0


def test_seg_log_softmax_and_stack_recovers_weights():
    rng = np.random.default_rng(0)
    spec = np.repeat(np.arange(400), 6)
    starts = BL._starts(spec)
    a, b = rng.normal(size=len(spec)), rng.normal(size=len(spec))
    t = np.exp(BL.seg_log_softmax(1.5 * a - 0.5 * b, starts, spec))
    assert np.allclose(np.add.reduceat(t, starts), 1.0)
    w = BL.fit_stack([a, b], t, starts, spec)
    assert w == pytest.approx([1.5, -0.5], abs=1e-3)


def test_spectrum_metrics_pair_classes_and_tv():
    labels = np.array(["C-O", "C-O", "C-C", BL.PRECURSOR])
    t = np.array([0.5, 0.1, 0.4, 0.0])
    right = np.log(np.array([0.5, 0.1, 0.4, 1e-9]) / (1 + 1e-9))
    swapped = np.log(np.array([0.1, 0.5, 0.4, 1e-9]) / (1 + 1e-9))  # wrong C-O bond only
    m = BL.spectrum_metrics({"r": right, "s": swapped}, t, labels, np.array([0]))
    agg = BL.aggregate(m, ["r", "s"])
    assert agg["r"]["acc_cross_type"] == 1.0 and agg["r"]["tv_bond"] == pytest.approx(0.0, abs=1e-6)
    assert agg["s"]["acc_same_type"] == 0.0 and agg["s"]["tv_type"] == pytest.approx(0.0, abs=1e-6)
    assert agg["s"]["tv_within"] == pytest.approx(0.4, abs=1e-6)
    # Cross-type pairs (0.5 vs 0.4) and (0.1 vs 0.4) both flip when the C-O masses swap.
    assert agg["s"]["acc_cross_type"] == 0.0
    # A tie scores half.
    tie = np.log(np.array([0.3, 0.3, 0.4, 1e-9]))
    tied = BL.aggregate(BL.spectrum_metrics({"x": tie}, t, labels, np.array([0])), ["x"])
    assert tied["x"]["acc_same_type"] == 0.5


def _toy_peak_recs():
    import pandas as pd

    # Spectrum 0: bonds 0, 1 and the precursor; fragments: precursor, cut at 0, cut at {0, 1}.
    # Fragments 1 and 2 share m/z group 1 (isomers), so only their sum is identifiable.
    r0 = {
        "bond_idx": np.array([0, 1, -1], dtype=np.int32),
        "frag_bond_flat": np.array([0, 0, 1], dtype=np.int16),
        "frag_bond_ptr": np.array([0, 0, 1, 3], dtype=np.int32),
        "chan_frag": np.array([0, 0, 1, 2], dtype=np.int16),
        "chan_group": np.array([0, 2, 1, 1], dtype=np.int16),
        "chan_w": np.array([0.3, 0.1, 0.4, 0.2], dtype=np.float32),
        "group_t": np.array([0.2, 0.7, 0.0], dtype=np.float32),
        "unc_t2": 0.01,
        "nce": 15.0,
    }
    r1 = {
        "bond_idx": np.array([2, -1], dtype=np.int32),
        "frag_bond_flat": np.array([2], dtype=np.int16),
        "frag_bond_ptr": np.array([0, 1, 1], dtype=np.int32),
        "chan_frag": np.array([0, 1], dtype=np.int16),
        "chan_group": np.array([0, 1], dtype=np.int16),
        "chan_w": np.array([0.8, 0.2], dtype=np.float32),
        "group_t": np.array([0.5, 0.5], dtype=np.float32),
        "unc_t2": 0.0,
        "nce": 45.0,
    }
    return pd.DataFrame([r0, r1])


def test_peak_split_members_metrics_and_gradient():
    recs = _toy_peak_recs()
    peak = BL.PeakSplit(recs, np.array([0, 3]))
    # Fragment members (bond-table rows): precursor -> its row; cut bonds -> their rows.
    assert peak.members.tolist() == [2, 0, 0, 1, 3, 4]
    assert peak.member_n.tolist() == [1, 1, 2, 1, 1]
    score = np.array([1.0, 3.0, 5.0, 2.0, 4.0])
    assert peak.bond_mean(score).tolist() == [5.0, 1.0, 2.0, 2.0, 4.0]
    # Precursor flags and NCE bins per fragment, and the derived stack columns.
    assert peak.is_prec.tolist() == [1, 0, 0, 0, 1]
    assert peak.nce_bin.tolist() == [0, 0, 0, 2, 2]
    base = {"B": peak.bond_mean(score)}
    assert peak.column(base, "B|prec").tolist() == [5.0, 0, 0, 0, 4.0]
    assert peak.column(base, "B|bond").tolist() == [0, 1.0, 2.0, 2.0, 0]
    assert peak.column(base, "prec@2").tolist() == [0, 0, 0, 0, 1.0]
    # Identity weights reproduce the model: spectrum 1 predicts (0.8, 0.2) vs target (0.5, 0.5).
    X = peak.log_wf[:, None]
    m = peak.metrics(X, np.array([1.0]))
    assert m.cos[1] == pytest.approx((0.8 * 0.5 + 0.2 * 0.5) / (np.hypot(0.8, 0.2) * np.hypot(0.5, 0.5)))
    assert m.tv[1] == pytest.approx(0.3)
    # Analytic gradient matches finite differences.
    X2 = np.stack([peak.log_wf, peak.bond_mean(score)], 1)
    theta = np.array([0.8, 0.3])
    _, g = peak.ce_and_grad(X2, theta)
    num = [(peak.ce_and_grad(X2, theta + e)[0] - peak.ce_and_grad(X2, theta - e)[0]) / 2e-6
           for e in np.eye(2) * 1e-6]
    assert g == pytest.approx(num, rel=1e-4)


def test_clustered_diff():
    d, se = BL.clustered_diff(np.array([1.0, 1.0, 3.0]), np.zeros(3), np.array(["a", "a", "b"]))
    assert d == pytest.approx(5 / 3)
    assert se == pytest.approx(np.sqrt((2 * (1 - 5 / 3)) ** 2 + (3 - 5 / 3) ** 2) / 3)
