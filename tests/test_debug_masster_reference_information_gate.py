"""Scientific controls for the frozen-prediction reference gate."""

import copy

import numpy as np
import pandas as pd
import pytest
import torch as th
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from fragnnet.utils.spec_similarity import cos_sim_helper
from fragnnet.utils.spec_utils import batched_bin_func
from scripts.debug.masster_memorization.debug_masster_reference_information_gate import (
    fit_correction,
    metric_cache,
    query_features,
    reference_features,
    scores,
)
from scripts.debug.masster_memorization.prepare_masster_reference_gate import retrieve_condition


def frame() -> pd.DataFrame:
    """Include a merged production bin, uncovered truth and out-of-range prediction."""
    result = pd.DataFrame(
        {
            "group": [0, 1, 2, 3],
            "mz": [10.0, 10.001, 20.02, 31.0],
            "native_mz": np.array([10.0, 10.001, 20.02, 31.0], dtype=np.float64),
            "pred": [0.1, 0.2, 0.3, 0.4],
            "target": [0.2, 0.0, 0.3, 0.0],
            "eligible": [True, True, True, False],
            "ion": ["H2O1", "H3N1", "C2H4", "C3H7"],
            "loss": ["C1", "C2", "C3", ""],
            "structural_alternatives": [1, 2, 3, 1],
            "formula_alternatives": [1, 1, 1, 1],
        }
    )
    result.attrs.update(
        true_mzs=np.array([10.0, 20.02, 25.0], dtype=np.float32),
        true_ints=np.array([0.2, 0.3, 0.5], dtype=np.float32),
    )
    return result


def test_metric_matches_production_with_native_dtypes_and_uncovered_truth():
    data = frame()
    pred = th.tensor(data.pred.to_numpy()[:3])
    pm = th.tensor(data.native_mz.to_numpy()[:3])
    tm, ti = th.from_numpy(data.attrs["true_mzs"]), th.from_numpy(data.attrs["true_ints"])
    target = batched_bin_func(tm, ti, th.zeros(3, dtype=th.long), 30.0, 0.01, "sum", sparse=True)
    predicted = batched_bin_func(
        pm, pred, th.zeros(3, dtype=th.long), 30.0, 0.01, "sum", sparse=True
    )
    expected = cos_sim_helper(*target, *predicted).item()
    cosine, top, rank = scores(np.log(data.pred.to_numpy()), metric_cache(data, 30.0))
    assert cosine == pytest.approx(expected, abs=1e-7)
    assert top == 0  # Uncovered true base peak cannot become a correct covered base peak.
    assert rank == 0
    assert cosine < 0.8


def test_reference_shuffle_preserves_values_and_masks_and_structure_ignores_labels():
    data = frame()
    row = pd.Series(
        {"tanimoto": 0.3, "query_prec_mz": 60, "donor_prec_mz": 63, "nce": 60, "inst_type": "ORB"}
    )
    st, labels, shuffled, mask = reference_features(data, data, row, "ion", 1e-6, 42)
    assert mask.tolist() == [1, 1, 1, 0]
    assert sorted(labels[:3, 0]) == sorted(shuffled[:3, 0])
    assert np.all(labels[:3, 0] != shuffled[:3, 0])
    assert np.all(labels[3] == 0)
    altered = data.copy()
    altered["target"] = [0.9, 0.8, 0.7, 0.6]
    new_st, _, _, new_mask = reference_features(altered, altered, row, "ion", 1e-6, 42)
    np.testing.assert_array_equal(st, new_st)
    np.testing.assert_array_equal(mask, new_mask)
    np.testing.assert_array_equal(
        query_features(data, row, 1e-6), query_features(altered, row, 1e-6)
    )


def test_fit_does_not_use_held_labels_and_unmatched_logits_stay_fixed():
    cfg = {
        "hidden_dim": 4,
        "learning_rate": 0.01,
        "weight_decay": 0.001,
        "epochs": 3,
        "patience": 3,
        "batch_spectra": 2,
    }
    data = []
    for _ in range(3):
        data.append(
            {
                "x": np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32),
                "x_mask": np.array([1, 1, 1, 0], dtype=np.float32),
                "target": frame().target.to_numpy(),
                "metric": metric_cache(frame(), 30.0),
            }
        )
    base = [np.log(frame().pred.to_numpy()) for _ in data]
    first, state = fit_correction(data, [0], [1], "x", base, cfg, 4)
    changed = copy.deepcopy(data)
    changed[2]["target"] = np.array([0.0, 0.0, 0.0, 1.0])
    changed[2]["metric"]["true_at_pred"] *= 0
    second, other = fit_correction(changed, [0], [1], "x", base, cfg, 4)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
        assert a[3] == pytest.approx(base[0][3], abs=1e-7)
    assert state["epoch"] == other["epoch"]
    assert state["fit"] == [0] and state["stop"] == [1]


def test_retrieval_excludes_identity_formula_and_matches_mass_distance():
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = {i: generator.GetFingerprint(Chem.MolFromSmiles("C" * (i + 1))) for i in range(10)}
    query = pd.Series({"mol_id": 0, "prec_mz": 100.0, "identity": "query", "formula": "Q"})
    pool = pd.DataFrame(
        [
            {"mol_id": i, "prec_mz": 101.0, "identity": str(i), "formula": str(i)}
            for i in range(1, 10)
        ]
    )
    pool.loc[0, "identity"] = "query"
    pool.loc[1, "formula"] = "Q"
    cfg = {
        "mass_window_da": 10.0,
        "mass_window_fraction": 0.05,
        "control_mass_distance_tolerance_da": 2.0,
    }
    neighbor, control, _ = retrieve_condition(query, pool, fps, cfg, np.random.default_rng(42))
    assert neighbor.mol_id not in (1, 2) and control.mol_id not in (1, 2)
    assert neighbor.mol_id != control.mol_id
    assert abs(abs(control.prec_mz - 100) - abs(neighbor.prec_mz - 100)) <= 2
