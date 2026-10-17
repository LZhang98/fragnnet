import argparse
import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import torch as th

from fragnnet.utils.spec_similarity import cos_sim_helper
from fragnnet.utils.spec_utils import batched_bin_func

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "debug" / "masster_memorization" / "matched_intensity_filter.py"
SPEC = importlib.util.spec_from_file_location("matched_intensity_filter", SCRIPT_PATH)
MIF = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MIF)


@pytest.fixture
def glycine_like() -> tuple[np.ndarray, np.ndarray]:
    """A small FT-like spectrum: base peak, a mid peak, and a weak peak."""
    return np.array([30.0344, 48.0444, 76.0393]), np.array([1000.0, 250.0, 4.0])


def test_pred_filter_keeps_peak_whose_candidates_sum_past_threshold():
    # Two candidates of 4 at one m/z form a peak of 8, above 0.5 * 10; each alone is below.
    mzs = np.array([200.0, 150.0, 150.0])
    ints = np.array([10.0, 4.0, 4.0])
    kept_mzs, kept_ints = MIF.pred_peak_filter(mzs, ints, 0.5)
    assert kept_mzs.tolist() == [200.0, 150.0, 150.0]
    assert kept_ints.sum() == pytest.approx(18.0)
    # At 0.85 the aggregated 8 falls below 0.85 * 10 and both candidates go.
    kept_mzs, _ = MIF.pred_peak_filter(mzs, ints, 0.85)
    assert kept_mzs.tolist() == [200.0]


def test_target_filter_is_strict_and_drops_above_mz_max():
    mzs = np.array([100.0, 200.0, 300.0, 1600.0])
    ints = np.array([100.0, 1.0, 1.5, 50.0])
    kept_mzs, kept_ints = MIF.target_filter(mzs, ints, 0.01)
    # 1.0 is not strictly above 0.01 * 100; 1600 is past mz_max.
    assert kept_mzs.tolist() == [100.0, 300.0]
    assert kept_ints.tolist() == [100.0, 1.5]


def test_zero_threshold_keeps_every_positive_peak(glycine_like):
    mzs, ints = glycine_like
    for fn in (MIF.pred_peak_filter, MIF.target_filter):
        kept_mzs, kept_ints = fn(mzs, ints, 0.0)
        np.testing.assert_array_equal(kept_mzs, mzs)
        np.testing.assert_array_equal(kept_ints, ints)


def test_cos_binned_matches_repo_bin_and_cosine():
    rng = np.random.default_rng(0)
    for _ in range(20):
        tm = np.sort(rng.uniform(50, 600, 15))
        ti = rng.uniform(1, 100, 15)
        # Predictions: some true peaks shifted within a bin, plus decoys.
        pm = np.concatenate([tm[:8] + rng.normal(0, 1e-4, 8), rng.uniform(50, 600, 10)])
        pi = rng.uniform(1, 100, len(pm))
        ours = MIF.cos_binned(pm, pi, tm, ti)
        t_bins = batched_bin_func(th.as_tensor(tm), th.as_tensor(ti), th.zeros(15, dtype=th.long),
                                  1500.0, 0.01, "sum", sparse=True)
        p_bins = batched_bin_func(th.as_tensor(pm), th.as_tensor(pi),
                                  th.zeros(len(pm), dtype=th.long), 1500.0, 0.01, "sum",
                                  sparse=True)
        ref = float(cos_sim_helper(*t_bins, *p_bins)[0])
        assert ours == pytest.approx(ref, abs=1e-9)


def test_cos_hungarian_sums_candidates_at_one_mz(glycine_like):
    tm, ti = glycine_like
    assert MIF.cos_hungarian(tm, ti, tm, ti) == pytest.approx(1.0)
    # The base peak split into two equal candidates is the same spectrum once summed.
    pm = np.array([tm[0], tm[0], tm[1], tm[2]])
    pi = np.array([500.0, 500.0, 250.0, 4.0])
    assert MIF.cos_hungarian(pm, pi, tm, ti) == pytest.approx(1.0)


def test_cos_hungarian_tolerance_and_disjoint():
    tm, ti = np.array([500.0]), np.array([1.0])
    # FT tolerance at 500 Da is 1e-5 * 500 = 0.005.
    assert MIF.cos_hungarian(np.array([500.004]), np.array([1.0]), tm, ti) == pytest.approx(1.0)
    assert MIF.cos_hungarian(np.array([500.006]), np.array([1.0]), tm, ti) == 0.0


def test_export_filters_both_sides_and_rebuilds_index(tmp_path):
    # Spectrum 7: a base peak, two 0.06% candidates summing to 0.12% (kept at 0.1%), one 0.05%
    # candidate (dropped). Spectrum 9: two peaks, both kept.
    pred_fp = tmp_path / "predictions_test.hdf5"
    with h5py.File(pred_fp, "w") as f:
        f["spec_id"] = np.array([7, 9])
        f["mol_id"] = np.array([1, 2])
        f["start_points"] = np.array([0, 4])
        f["end_points"] = np.array([4, 6])
        f["pred_mzs"] = np.array([100.0, 50.0, 50.0, 60.0, 80.0, 90.0])
        f["pred_ints"] = np.array([1000.0, 0.6, 0.6, 0.5, 1000.0, 400.0], dtype=np.float32)
    scores = pd.DataFrame({
        "spec_id": [7, 9, 7, 9], "mode": ["none", "none", "both", "both"],
        "thresh": [0.0, 0.0, 0.001, 0.001], "cos_hun": [0.5, 0.6, 0.51, 0.61],
        "cos_sim": [0.4, 0.5, 0.41, 0.51],
    })
    scores.to_pickle(tmp_path / "scores.pkl")
    targets = pd.DataFrame({"spec_id": [7, 9], "peaks": [
        [(100.0, 1000.0), (50.0, 0.5), (1600.0, 900.0)], [(80.0, 10.0), (90.0, 5.0)]]})
    targets.to_pickle(tmp_path / "targets.pkl")
    spec_df = pd.DataFrame({"spec_id": [7, 9, 11], "nce": [30.0, 40.0, 50.0],
                            "peaks": [None, None, None]})
    spec_df.to_pickle(tmp_path / "spec_df.pkl")

    MIF.run_export(argparse.Namespace(
        pred_hdf5=str(pred_fp), scores_fp=str(tmp_path / "scores.pkl"),
        targets_fp=str(tmp_path / "targets.pkl"), spec_df_fp=str(tmp_path / "spec_df.pkl"),
        thresh=0.001, out_dp=str(tmp_path / "out")))

    with h5py.File(tmp_path / "out" / "predictions_test.hdf5", "r") as f:
        # The two m/z 50 candidates become one peak of 1.2; peaks are m/z-sorted, base = 1000.
        assert f["pred_mzs"][:].tolist() == [50.0, 100.0, 80.0, 90.0]
        np.testing.assert_allclose(f["pred_ints"][:], [1.2, 1000.0, 1000.0, 400.0], rtol=1e-6)
        assert f["start_points"][:].tolist() == [0, 2]
        assert f["end_points"][:].tolist() == [2, 4]
        assert f["cos_hun"][:].tolist() == pytest.approx([0.51, 0.61])
        assert f["cos_hun_inst_tol"][:].tolist() == pytest.approx([0.51, 0.61])
        assert f["mol_id"][:].tolist() == [1, 2]
    out_df = pd.read_pickle(tmp_path / "out" / "spec_df_test.pkl").set_index("spec_id")
    # 0.5 is not above 0.001 * 1000; 1600 is past mz_max; spectrum 11 was not exported.
    assert out_df.index.tolist() == [7, 9]
    assert out_df.loc[7, "peaks"] == [(100.0, 1000.0)]
    assert out_df.loc[9, "peaks"] == [(80.0, 10.0), (90.0, 5.0)]
